import os, sys
import logging
sys.path.append('../')
from data.metro.preprocessor import preprocessing
from data.metro.metro import load_data
from models.STResNet import STResNet
import torch
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu = torch.device("cuda:0")

def predict(in_data,out_data,start_time):
    """
    :param io_data: 数据库trips中从start_time开始6个小时进出站的数据,[进站数据，出站数据]
    :param start_time: 时间片开始的时间，必须是以下格式中的一种[xxxx-xx-xx 00:00:00,xxxx-xx-xx 06:00:00,xxxx-xx-xx 12:00:00,xxxx-xx-xx 18:00:00]
    :return: data_out: 预测的数据
    """
    logging.basicConfig(level=logging.DEBUG, format=' %(levelname)s - %(message)s')
    nb_epoch = 500  # number of epoch at training stage
    batch_size = 32  # batch size
    T = 4  # number of time intervals in one day
    len_closeness = 3  # length of closeness dependent sequence
    len_period = 1  # length of  peroid dependent sequence
    len_trend = 1  # length of trend dependent sequence
    days_test = 7 * 4
    len_test = T * days_test
    map_height, map_width = 31, 20  # grid size
    nb_flow = 2
    lr = 0.0002  # learning rate
    nb_residual_unit = 4

    data,date = preprocessing(in_data,out_data,start_time)
    data_list,external_dim,mmn = load_data(data=data,date=date,len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)

    model = STResNet(
        learning_rate=lr,
        epoches=nb_epoch,
        batch_size=batch_size,
        len_closeness=len_closeness,
        len_trend=len_trend,
        external_dim=external_dim,
        map_heigh=map_height,
        map_width=map_width,
        nb_flow=nb_flow,
        nb_residual_unit=nb_residual_unit,
    )

    if gpu_available:
        print("==============")
        model = model.to(gpu)
    model.load_model("model.pt")
    # data_out = model.predict(torch.Tensor(data_in))

    data_in = []
    for data_t in data_list:
        data_in.append(torch.Tensor(data_t))
    data_out = model.predict(data_in)
    return data_out
