# Acknowledgement: This code is taken from https://github.com/TolicWang/DeepST
import numpy as np
import pandas as pd
from .timestamp import string2timestamp
import datetime


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        x = len(data)
        # timestamps = np.array(timestamps)
        # assert data.shape[0] == timestamps.shape[0]
        self.data = data
        self.timestamps = timestamps# [b'2019-04-20 00:00:00']
        self.T = T
        self.pd_timestamps = timestamps
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()  # 将时间戳：做成一个字典，也就是给每个时间戳一个序号

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):  # 给定时间戳返回对于的数据
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version

        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 时间偏移 minutes = 30
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        # print depends # [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        t = datetime.datetime.strptime(self.pd_timestamps[0], "%Y-%m-%d %H:%M:%S")
        x_c = self.data[:3]
        # 取当前时刻的前3个时间片的数据数据构成“邻近性”模块中一个输入序列
        # 例如当前时刻为[Timestamp('2013-07-01 00:00:00')]
        # 则取：
        # [Timestamp('2013-06-30 23:30:00'), Timestamp('2013-06-30 23:00:00'), Timestamp('2013-06-30 22:30:00')]
        #  三个时刻所对应的in-out flow为一个序列
        x_p = self.data[3]
        # 取当前时刻 前 1*PeriodInterval,2*PeriodInterval,...,len_period*PeriodInterval
        # 天对应时刻的in-out flow 作为一个序列，例如按默认值为 取前1、2、3天同一时刻的In-out flow
        x_t = self.data[4]
        # 取当前时刻 前 1*TrendInterval,2*TrendInterval,...,len_trend*TrendInterval
        # 天对应时刻的in-out flow 作为一个序列,例如按默认值为 取 前7、14、21天同一时刻的In-out flow
        if len_closeness > 0:
            XC.append(np.vstack(x_c))
            # a.shape=[2,32,32] b.shape=[2,32,32] c=np.vstack((a,b)) -->c.shape = [4,32,32]
        XP = x_p[np.newaxis,:]
        XT = x_t[np.newaxis,:]
        timestamps_Y.append(self.timestamps[0])#[]

        XC = np.asarray(XC)  # 模拟 邻近性的 数据 [?,6,32,32]
        XP = np.asarray(XP)  # 模拟 周期性的 数据 隔天
        XT = np.asarray(XT)  # 模拟 趋势性的 数据 隔周

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape)
        return XC, XP, XT, timestamps_Y


if __name__ == '__main__':
    # depends = [range(1, 3 + 1),
    #            [1 * 48 * j for j in range(1, 3 + 1)],
    #            [7 * 48 * j for j in range(1, 3 + 1)]]
    # print(depends)
    # print([j for j in depends[0]])
    str = ['2013070101']
    t = string2timestamp(str)
    offset_frame = pd.DateOffset(minutes=24 * 60 // 48)  # 时间偏移 minutes = 30
    print(t)
    o = [t[0] - j * offset_frame for j in range(1, 4)]
    print(o)
