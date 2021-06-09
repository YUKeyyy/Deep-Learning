import torchvision
from skimage import io,transform
import os
import pickle
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms


class MyDatasets(data.Dataset):#继承data.Dataset
    #初始化函数
    def __init__(self,Input_root_dir,Target_root_dir,transform=None):

        #输入数据文件目录
        self.Input_root_dir = Input_root_dir
        # 目标数据文件目录
        self.Target_root_dir = Target_root_dir
        #变换
        self.transform = transform
        #目录里的所有文件
        self.Inputdata = os.listdir(self.Input_root_dir)
        self.Targetdata = os.listdir(self.Target_root_dir)

    #返回整个输入数据集的大小
    def __len__(self):
        return len(self.Inputdata)

    #根据索引返回对应文件
    def __getitem__(self, index):
        #根据索引item获取该输入数据
        data_index = self.Inputdata[index]

        #返回数据路径
        data_path = os.path.join(self.Input_root_dir,data_index)

        #读取数据
        data_input = (np.load(data_path,allow_pickle=True))
        data_input = data_input[0:256,0:128]

        noiseLevel , dataIndex = data_index.split('.')[0].split('_')[-1].split('NoiseData')

        data_target_name = 'theoreticalData'+str(dataIndex)+'.npy'

        data_target = (np.load(os.path.join(self.Target_root_dir,data_target_name),allow_pickle=True))
        data_target = data_target[0:256,0:128]

        if self.transform:
            data_input = self.transform(data_input)
            data_target = self.transform(data_target)

        data_input.unsqueeze(0)
        data_target.unsqueeze(0)

        return data_input,data_target,data_target_name,noiseLevel

if __name__ == '__main__':


    Input_root_dir = r'Data\Numpy_DATA\AddNoise'
    Target_root_dir =r'Data\Numpy_DATA\Original'
    data = MyDatasets(Input_root_dir=Input_root_dir,Target_root_dir=Target_root_dir,transform=None)
    dataloader = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)

    input_example, target_example,data_target_name,noiseLevel = next(iter(dataloader))
    print(u'X_example:{}'.format((input_example)))
    print(u'Y_example:{}'.format((target_example)))
    print(data_target_name)
    print(noiseLevel)



