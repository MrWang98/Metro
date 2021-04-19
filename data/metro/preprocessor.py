import pymysql
import datetime
import numpy as np
import copy


def preprocessing(in_data,out_data,start_time):
    """
    :param in_data: 数据库中前3个6小时时间片,一天前的同一个时刻的6小时时间片，前一周同一时刻6小时的时间片的进站数据
    :param out_data: 数据库中前3个6小时时间片,一天前的同一个时刻的6小时时间片，前一周同一时刻6小时的时间片的出站数据
    :param start_time: 时间片开始的时间，必须是以下格式中的一种[xxxx-xx-xx 00:00:00,xxxx-xx-xx 06:00:00,xxxx-xx-xx 12:00:00,xxxx-xx-xx 18:00:00]
    :return: 处理后可以输入模型预测的数据
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
        first_line.append(station[0])
        stations[station[0]] = 0
    station_location = np.load("D:\编程\服创\model\data\metro\station_location.npy")
    d1,d2 = station_location.shape

    startArray = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    date = []
    date.append(datetime.datetime.strftime(startArray - datetime.timedelta(hours=6), "%Y-%m-%d %H:%M:%S"))
    date.append(datetime.datetime.strftime(startArray - datetime.timedelta(hours=12), "%Y-%m-%d %H:%M:%S"))
    date.append(datetime.datetime.strftime(startArray - datetime.timedelta(hours=18), "%Y-%m-%d %H:%M:%S"))
    date.append(datetime.datetime.strftime(startArray - datetime.timedelta(hours=24), "%Y-%m-%d %H:%M:%S"))
    date.append(datetime.datetime.strftime(startArray - datetime.timedelta(hours=24 * 7), "%Y-%m-%d %H:%M:%S"))


    #将进站数据处理成时间片
    in_flow = []
    for data_all in in_data:
        temp = copy.deepcopy(stations)
        for data_t in data_all:
            if data_t[0] not in stations:
                print(data_t[0])
            else:
                temp[data_t[0]]=data_t[1]
        in_flow_temp = []
        for i in range(d1):
            line = []
            for j in range(d2):
                station = copy.deepcopy(station_location[i][j])
                if station == 'Sta0':
                    line.append(0)
                else:
                    if station not in stations:
                        line.append(0)
                    else:
                        line.append(temp[station])
            in_flow_temp.append(line)
        in_flow.append(in_flow_temp)

    #将出站顺序处理成时间片
    out_flow = []
    for data_all in out_data:
        temp = copy.deepcopy(stations)
        for data_t in data_all:
            if data_t[0] not in stations:
                print(data_t[0])
            else:
                temp[data_t[0]] = data_t[1]
        out_flow_temp = []
        for i in range(d1):
            line = []
            for j in range(d2):
                station = copy.deepcopy(station_location[i][j])
                if station == 'Sta0':
                    line.append(0)
                else:
                    if station not in stations:
                        line.append(0)
                    else:
                        line.append(temp[station])
            out_flow_temp.append(line)
        out_flow.append(out_flow_temp)

    allData = []
    for data1,data2 in zip(in_flow,out_flow):
        allData.append([data1,data2])
    # date = datetime.datetime.strptime(start_time,"%Y-%m-%d %H:%M:%S")
    return allData,date