import h5py
import torch
import visdom
import scipy.io as sio
import MainCode.TOOL.ToolClass as tool
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D



def normalization(data):
    _range = np.max(abs(data))
    return data / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
#
# data_path = r'Data\Numpy_DATA\Origin\theoreticalData1.npy'
# # 读取数据
# data_input = (np.load(data_path))[0:1280, 0:192]
# nps = data_input
# print(data_input.shape)
#
# #1500*200:nps
#
# Dao_num = nps.shape[0]
# Points_num = nps.shape[1]
# for i in range(Points_num):
#     nps[:,i] = normalization(nps[:,i])

# block = []
# block.insert(-1,5)
# block.insert(-2,56)
# print(block)

print(list(range(1,5)))

# print(nps)
# TODO:制作3D图像实验

# vis = visdom.Visdom(env=u'test', use_incoming_socket=False)
# X = np.zeros([Dao_num,Points_num])
# for i in range(0,Dao_num):
#     X[i] = np.arange(0,Points_num,1)


# fig = figure()
# ax = Axes3D(fig)
# x = np.arange(0,Dao_num,1)
# y = np.arange(0,Points_num,1)
# [x,y] = np.meshgrid(x,y) # 转换成二维的矩阵坐标
# z = nps[x,y]
#
# ax.plot_surface(x,y,z,rstride=1,cstride=2,cmap='winter')
# show()

# print(X.shape)
#
# N= np.linspace(0,Points_num,Points_num)
# N = np.insert(N,N.shape[0])
#N = np.column_stack((N,N))
#N = np.transpose(N)
# print(N.shape)
# print(nps[0:2].shape)

#nps[1]+=1
# vis.line(X=N,Y=nps[0:2],
#          opts=dict(
#                    showlegend=True,
#                    markers=False,
#                    title='line demo',
#                    xlabel='Time',
#                    ylabel='Volume',
#                    fillarea=True),
#          )





    #  time.sleep(0.5)

# z: list = [0]
# y: list = [0]
# x: list = [0]
#
# fig = figure()
# ax = Axes3D(fig)
# for i in range(50):
#     z.append(randint(2,5))
#     x.append(i)
#     y.append(i)
#     ax.plot(x,y,z,color='blue')
#     plt.pause(0.1)
#     plt.ioff()


# ax = Axes3D(fig)
# x = np.arange(0,Dao_num,1)
# y = np.arange(0,Points_num,1)
# [x,y] = np.meshgrid(x,y) # 转换成二维的矩阵坐标
# z = nps[x,y]
#
# ax.plot_surface(x,y,z,rstride=1,cstride=2,cmap='winter')
# show()

# print(nps)

# nps = np.transpose(nps)
#
# print(nps.shape)
# tool.__Draw__(nps)
# plt.show()
#
# print(nps)