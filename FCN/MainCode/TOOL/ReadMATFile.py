import h5py
import scipy.io as sio
import MainCode.TOOL.ToolClass as tool
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


def FromMatToNumpy_ReadArray(targetPath,arrayName):
    """
    读取mat中的数组到python中，并以numpy数组返回；

    :param targerPath: 读取.mat文件的路径；
    :param arrayName: 读取数组的名称【一般是文件名】；
    :return: np类型的数组；
    """
    nps = None
    mat = sio.loadmat(targetPath)
    try:
        nps = (np.array(mat[arrayName]))
    except np.ERR_PRINT:
        print('ERROR  FromMatToNumpy_ReadArray :{}'.format(arrayName))
    return nps


def MKDIR_File(targetPath):
    '''
    简易检测是否存在文件夹，如果不存在就按照路径创建一个

    :param targetPath: 文件夹路径
    :return: NULL
    '''
    if not os.path.exists(targetPath):
        os.mkdir(targetPath)

if __name__ == '__main__':

    # TODO:将.mat文件转换为.npy文件，并划分含噪数据集和原始数据集

    src_path = '..\..\Data\MATLAB_DATA'
    All_files = os.listdir(src_path)
    len = len(All_files)
    for item in All_files:
        name = str(item).split('.')[0]
        if 'NoiseData' in name:
            save_path = r'..\..\Data\Numpy_DATA\AddNoise'
            MKDIR_File(save_path)
        else:
            save_path = r'..\..\Data\Numpy_DATA\Original'
            MKDIR_File(save_path)
        npArr = FromMatToNumpy_ReadArray(targetPath=os.path.join(src_path, item), arrayName=name)
        np.save(file=os.path.join(save_path, name), arr=npArr, allow_pickle=True, fix_imports=True)
        print(str(name))
