from Metro import predict
import pymysql
import datetime
import numpy as np
import os
import copy

def getData(start_time):
        """
        :param start_time: #要预测时间片开始时间的往前6个小时，例如要预测[2020-01-28 12:00:00]之后6个小时的流量，这个开始时间就是[2020-01-28 06:00:00]
        :return:
        """
        connect = pymysql.connect(
                host='host.tanhuiri.cn',
                port=3306,
                user='root',
                password='19991119+thr',
                db='metro',
                charset='utf8',
                use_unicode=True,
            )
        cur = connect.cursor()

        startArray = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        date = []
        date.append(datetime.datetime.strftime(startArray-datetime.timedelta(hours=6), "%Y-%m-%d %H:%M:%S"))
        date.append(datetime.datetime.strftime(startArray-datetime.timedelta(hours=12), "%Y-%m-%d %H:%M:%S"))
        date.append(datetime.datetime.strftime(startArray-datetime.timedelta(hours=18), "%Y-%m-%d %H:%M:%S"))
        date.append(datetime.datetime.strftime(startArray-datetime.timedelta(hours=24), "%Y-%m-%d %H:%M:%S"))
        date.append(datetime.datetime.strftime(startArray-datetime.timedelta(hours=24*7), "%Y-%m-%d %H:%M:%S"))

        in_data = []
        out_data = []

        #取一次测试的数据
        for str1 in date:
                str1Array = datetime.datetime.strptime(str1, "%Y-%m-%d %H:%M:%S")
                str2 = datetime.datetime.strftime(str1Array+datetime.timedelta(hours=6), "%Y-%m-%d %H:%M:%S")
                in_select_query = """
                            SELECT t.station_in,COUNT( t.id ) AS cnt
                            FROM
                                trips t
                            WHERE
                                t.time_in BETWEEN "{}" AND "{}" 
                            GROUP BY
                                t.station_in
                            ORDER BY
                                t.station_in""".format(str1,str2)

                out_select_query = """
                            SELECT t.station_out,COUNT( t.id ) AS cnt
                            FROM
                                trips t
                            WHERE
                                t.time_out BETWEEN "{}" AND "{}" 
                            GROUP BY
                                t.station_out
                            ORDER BY
                                t.station_out""".format(str1, str2)
                cur.execute(in_select_query)
                data1 = cur.fetchall()
                in_data.append(data1)
                cur.execute(out_select_query)
                data2 = cur.fetchall()
                out_data.append(data2)
        return in_data,out_data

if __name__ == '__main__':
        #要预测的时间片开始时间
        #且开始时间只能是[00:00:00、06:00:00、12:00:00、18:00:00]中的一个，被6整除的小时
        start_time = "2020-04-28 06:00:00"

        #从数据库中获得数据
        in_data,out_data = getData(start_time)

        #输入预测
        pred = predict(in_data,out_data,start_time)

        connect = pymysql.connect(
                host='host.tanhuiri.cn',
                port=3306,
                user='root',
                password='19991119+thr',
                db='metro',
                charset='utf8',
                use_unicode=True,
        )
        cur = connect.cursor()

        select_query = """
                        SELECT s.station_id
                        FROM
                            station s"""
        cur.execute(select_query)
        stataions_list = cur.fetchall()
        first_line = []
        first_line.append("start_time")
        stations = {}
        for station in stataions_list:
                stations[station[0]] = 0

        x = pred.numpy()[0]
        print(os.path.realpath(__file__))
        station_location = np.load("D:\编程\服创\model\data\metro\station_location.npy")

        result = []     #[in_flow,out_flow]
        d1,d2,d3 = x.shape
        for i in range(d1):
                result_t = copy.deepcopy(stations)
                for j in range(d3):
                        for idx,t in enumerate(x[i][j]):
                                if station_location[j][idx]!="Sta0":
                                        if round(t) < 0:
                                                result_t[station_location[j][idx]] = 0
                                        else:
                                                result_t[station_location[j][idx]] = round(t)
                result.append(result_t)
        #输出
        print(result)

