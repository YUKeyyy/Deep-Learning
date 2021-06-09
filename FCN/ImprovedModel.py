import time
import numpy as np
import torch
import torchvision
from pylab import *
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms
from Datasets import MyDatasets
import torch.optim as optim

def normalization(data):
    _range = np.max(abs(data))
    return data / _range


# 定义密集块子块-- Bottle-Neck
def conv_bottleneck(in_channels,out_channels):
    block = torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,padding=0, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False)
    )
    return block

# 定义密集块-- Dense-Block
class DenseBlock(torch.nn.Module):

    def __init__(self,num_layers,in_channels,growth_rate):
        '''
        初始化密集块网络

        :param num_layers: 密集块中bottleneck的层数
        :param in_channels: 当前密集块的输入通道数
        :param growth_rate: 密集块的增长率
        '''
        super(DenseBlock,self).__init__()
        block=[]
        channel = in_channels

        # 自定义生成密集块并封装到block中
        for i in range(num_layers):
            block.append(conv_bottleneck(channel,growth_rate))
            channel += growth_rate

        # 将密集块序列化成为网络的一部分
        self.DenseBlock_Net = torch.nn.ModuleList(block)

    def forward(self,x):
        #print(self.net)
        for layer in self.DenseBlock_Net:
            out = layer(x)
            # 将两个张量拼接在一起（按维数1拼接）
            x = torch.cat((x,out),dim=1)
        return x


# 定义过渡层-- Transition-Block
def transition_block(in_channels, out_channels):
    '''
    定义过渡层

    :param in_channels:输入通道数
    :param out_channels: 输出通道数
    :return: 返回过渡层网络
    '''
    block = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            torch.nn.AvgPool2d(kernel_size=2, stride=2))
    return block


class Densenet(torch.nn.Module):
    def __init__(self,in_channel,mid_channels=24,growth_rate=12,block_layers=[2,3]):
        '''
        密集网络---基本网络

        :param in_channel: 输入数据的通道数
        :param mid_channels: 将数据进行浅层特征提取的输出通道数
        :param growth_rate: 密集块的增长率
        :param block_layers: 密集块中bottleneck的层数
        '''
        super(Densenet,self).__init__()

        self.in_channel = in_channel
        self.num_denseblock = len(block_layers)
        self.net = torch.nn.Sequential()
        self.head = torch.nn.Sequential()

        # 对数据进行浅层特征提取
        head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=1,padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, mid_channels, kernel_size=3, stride=1, padding=1)
        )
        self.head.add_module('head',head)

        self.temp_adv = torch.nn.Sequential()
        self.pool = torch.nn.Sequential()

        channels = mid_channels
        block = []
        poolinglist = []
        out_channels = []
        in_channels = []
        temp_adv = []
        # 构建密集块并将其连接到
        for i, layers in enumerate(block_layers):
            in_channels.append(channels)

            self.net.add_module("DenseBlock_%d"%(i),DenseBlock(num_layers=layers,in_channels=channels, growth_rate=growth_rate))

            temp_adv_item = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=channels, out_channels=channels + layers * growth_rate, kernel_size=1,stride=1,padding=0, bias=False)
            )

            self.temp_adv.add_module("Resnet_%d" % (i),temp_adv_item)


            channels += layers * growth_rate

            PL = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1,
                                stride=1, padding=0, bias=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )

            self.pool.add_module("MaxPool_%d"%(i),PL)

            out_channels.append(channels)


    def forward(self,X):
        output = {}
        num = 0

        X = self.head(X)

        for index , (net_layer,temp_layer,pool_layer) in enumerate(zip(self.net,self.temp_adv,self.pool)):
            # print(index)
            # print('---------')
            # print(net_layer)
            # print('---------')
            # print(temp_layer)
            # print('---------')
            # print(pool_layer)
            # print('---------')

            temp = X
            X = net_layer(X)
            temp = temp_layer(temp)
            X += temp
            X = pool_layer(X)

            num+=1
            output['pooling%d' % (num)] = X

        # 对数据进行浅层的特征提取
        # print(self.net)
        # self.net(X)
        # print('----------------')
        # print(self.net[0])

        return output

# x = Densenet(1)
# test_X = Variable(torch.zeros((1,1,96,96)))
# y = x(test_X)

class Ipr_FCN(torch.nn.Module):
    def __init__(self,in_channels,mid_channels=256,growth_rate=12,block_layers=[2,3]):
        '''
        基于Densenet进行改进的FCN网络

        :param in_channels: 数据的输入通道数
        :param mid_channels: 将数据进行浅层特征提取的输出通道数
        :param growth_rate: 密集块的增长率
        :param block_layers: 密集块中bottleneck的层数
        '''
        super(Ipr_FCN,self).__init__()

        # 设置主干网络为Densenet
        self.prev_net = Densenet(in_channel=in_channels,mid_channels=mid_channels,growth_rate=growth_rate,block_layers=block_layers)

        self.mid_channels =mid_channels
        self.block_layers = block_layers
        self.in_channel = in_channels

        self.relu = torch.nn.ReLU(inplace=True)

        out_channels_list = []
        in_channels_list = []

        self.decovn_net = torch.nn.Sequential()
        self.head  =torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_channels,out_channels=self.in_channel,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(self.in_channel)
        )

        temp_channels = mid_channels

        # 将经过DenseBlock输入通道数和输出通道数分别放到两个列表中
        for num_layers in block_layers:
            in_channels_list.append(temp_channels)
            temp_channels += growth_rate*num_layers
            out_channels_list.append(temp_channels)

        self.out_channels = out_channels_list
        self.in_channels = in_channels_list

        # 根据密集块的数量设置反卷积层，对数据和通道数进行还原
        decovnlist = []


        for i in range(1,len(block_layers)+1):
            decovn = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(out_channels_list[-1 * i], in_channels_list[-1 * i], kernel_size=3, stride=2,padding=1, dilation=1, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(in_channels_list[-1 * i]),
            )
            self.decovn_net.add_module('decovn_%d'%(len(block_layers)+1-i),decovn)

        self.classifier = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)



    def forward(self,X):
        output = self.prev_net(X)
        output_list = []


        # print('---output---')
        for index,out_layer in enumerate(output):
            output_list.insert(index,output[out_layer])
            # print(output[out_layer].shape)

        # print('---decovn---')
        # print(self.out_channels)

        for index,decovn_layer in enumerate(self.decovn_net,1):
            score = output_list[-1*index]
            score = self.relu(decovn_layer(score))
            # print(score.shape)
            if index<len(self.block_layers):
                score += output_list[-1*(index+1)]


        # 把数据的通道数回归至原始通道数
        score = self.head(score)
        score = self.classifier(score)
        # print(score.shape)
        return score


if __name__ == '__main__':

    model = Ipr_FCN(1, block_layers=[6,12,18])
    # print(model)

    # writer = SummaryWriter(comment='model')
    # fake = torch.randn(1,1,1280,192)
    # writer.add_graph(model=model,input_to_model=fake)
    # writer.close()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    if model.cuda():
        model = model.to(device)

    # print(model)

    batch_size = 1

    # TODO:测试版本，用数据去训练网络

    Input_root_dir = r'Data\Numpy_DATA\AddNoise'
    Target_root_dir = r'Data\Numpy_DATA\Original'
    data_trans = transforms.Compose([transforms.ToTensor()])
    data = MyDatasets(Input_root_dir=Input_root_dir, Target_root_dir=Target_root_dir, transform=data_trans)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    lr = 1e-2
    loss_f = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)

    # 设置训练次数
    train_n = 5

    time_open = time.time()

    step = 0

    fig = figure()
    ax = Axes3D(fig)

    for epoch in range(train_n):

        start_epoch = True

        print('epoch : ' + str(epoch))

        # 开启训练
        model.train()
        # 初始化loss和corrects
        running_loss = 0.0
        running_corrects = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch, (Input, Target, Name, NoiseLevel) in enumerate(dataloader, 1):
            # 将数据放在GPU上训练
            X, Y = Variable(Input).to(device), Variable(Target).to(device)
            X = X.type(torch.float32)
            Y = Y.type(torch.float32)

            # print(X.shape)
            # print(Y.shape)

            #print(X.shape)
            # X, Y = torch.Tensor(Variable(X)), Variable(Y)
            # 模型预测概率
            y_pred = model(X)
            # pred，概率较大值对应的索引值，可看做预测结果，1表示行
            _, pred = torch.max(y_pred.data, 1)
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_f(y_pred, Y)

            # 反向传播
            loss.backward()
            optimizer.step()





            # print('imag_narry:'+str(imag_narry))
            # print('out:'+str(out.shape))

            # 损失和
            running_loss += loss.data.item()

            # 预测正确的图片个数
            running_corrects += torch.sum(pred == Y.data)

            # 输出每个epoch的loss和acc[平均损失]
            epoch_loss = running_loss * batch_size / len(data)
            epoch_acc = 100 * running_corrects / len(data)
            # writer_array.add_scalar('batch loss',loss.data.item(),global_step=step)
            step += 1
            #print('\rlabel:{}\npred:{}\nLoss:{:.4f} Acc:{:.4f}%'.format(Y.data, pred.data, epoch_loss, epoch_acc),end='', flush=True)
            print('\rLoss:{:.4f} Acc:{:.4f}% step:{}'.format(epoch_loss, epoch_acc,step), end='', flush=True)

        scheduler.step()

        # TODO：开启测试
        # model.eval()
        #
        # data_path = r'Data\Numpy_DATA\test\theoretical_1NoiseData1.npy'
        # # 读取数据
        # data_input = (np.load(data_path))[0:1280, 0:192]
        # data_input = data_trans(data_input)
        # data_input = data_input.unsqueeze(0)
        # data_input = (Variable(data_input).to(device)).type(torch.float32)
        # print(data_input.shape)
        # # 模型预测概率
        # y_pred = model(data_input)
        # # pred，概率较大值对应的索引值，可看做预测结果，1表示行
        # _, pred = torch.max(y_pred.data, 1)
        #
        # out = y_pred.squeeze(0)
        # if start_epoch:
        #     imag_narry = (out.squeeze(0)).cpu().detach().numpy()
        #     Dao_num = imag_narry.shape[0]
        #     Points_num = imag_narry.shape[1]
        #     for i in range(Points_num):
        #         imag_narry[:, i] = normalization(imag_narry[:, i])
        #     x = np.arange(0, Dao_num, 1)
        #     y = np.arange(0, Points_num, 1)
        #     [x, y] = np.meshgrid(x, y)  # 转换成二维的矩阵坐标
        #     z = imag_narry[x, y]
        #
        #     # plt.clf()  # 清除之前画的图
        #     ax.plot_surface(x, y, z, rstride=1, cstride=2, cmap='winter')
        #     plt.pause(10)
        #     plt.ioff()
        #     start_epoch = False

        # writer_array.add_scalar('epoch loss', epoch_loss, global_step=epoch)

        print('\n')

    time_end = time.time() - time_open
    print(time_end)
