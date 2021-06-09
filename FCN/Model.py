import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from pylab import *
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from torchvision import transforms
from Datasets import MyDatasets
import torch.optim as optim
from tensorboardX import SummaryWriter
# model = torchvision.models.vgg16()
# del model.classifier
# print(model)
# print('\n')
# model = torchvision.models.resnet18()
# print(model)


# ranges 是用于方便获取和记录每个池化层得到的特征图
# 例如vgg16，需要(0, 5)的原因是为方便记录第一个pooling层得到的输出(详见下午、稳VGG定义)
ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg网络结构配置（数字代表经过卷积后的channel数，‘M’代表卷积层）
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 由cfg构建Vgg-net的卷积层和池化层
def Make_layers(cfg, batch_normal=False):
    layers = []
    # 输入层的通道数
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_normal:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# m = Make_layers(cfg['vgg16'])

# class VGGnet(torchvision.models.VGG):
#     def __init__(self,pretrained = True,model='vgg16',requires_grad=True,remove_fc=True):
#         super(VGGnet,self).__init__(Make_layers(cfg[model]))
#         self.range = ranges[model]


class VGGnet(nn.Module):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True):
        super(VGGnet, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.layer = Make_layers(cfg[model])
        self.ranges = ranges[model]

    def forward(self, x):
        output = {}
        # 利用之前定义的ranges获取每个maxpooling层输出的特征图
        for idx, (begin, end) in enumerate(self.ranges):
            for layer in range(begin, end):
                x = self.features[layer](x)
                output["x%d" % (idx + 1)] = x
        # output 为一个字典键x1d对应第一个maxpooling输出的特征图，x2...x5类推
        return output

print(VGGnet())


class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super(FCN8s, self).__init__()

        self.n_class = n_class

        self.pretrained_net = pretrained_net

        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)

        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output['x5']  # maxpooling5的feature map (1/32)
        x4 = output['x4']  # maxpooling4的feature map (1/16)
        x3 = output['x3']  # maxpooling3的feature map (1/8)

        score = self.relu(self.conv6(x5))  # conv6  size不变 (1/32)
        #print('scorex5:'+str(score.shape))
        score = self.relu(self.conv7(score))  # conv7  size不变 (1/32)
        #print('scorex5relu:' + str(score.shape))
        score = self.relu(self.deconv1(x5))  # out_size = 2*in_size (1/16)
        #print(score.shape)
        #print(x4.shape)
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))  # out_size = 2*in_size (1/8)

        #print(score.shape)
        #print(x3.shape)

        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  # out_size = 2*in_size (1/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # out_size = 2*in_size (1/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # out_size = 2*in_size (1)
        score = self.classifier(score)  # size不变，使输出的channel等于类别数

        return score

def normalization(data):
    _range = np.max(abs(data))
    return data / _range



if __name__ == '__main__':
    vgg_model = VGGnet()
    model = FCN8s(vgg_model, 1)
    print(model)

    # writer = SummaryWriter(comment='model')
    # fake = torch.randn(1,1,1280,192)
    # writer.add_graph(model=model,input_to_model=fake)
    # writer.close()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    if model.cuda():
        model = model.to(device)

    #print(model)

    batch_size = 1

    # TODO:测试版本，用数据去训练网络

    Input_root_dir = r'Data\Numpy_DATA\AddNoise'
    Target_root_dir = r'Data\Numpy_DATA\Original'
    data_trans = transforms.Compose([transforms.ToTensor()])
    data = MyDatasets(Input_root_dir=Input_root_dir, Target_root_dir=Target_root_dir, transform=data_trans)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    lr = 0.0001
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
        model.eval()

        data_path = r'Data\Numpy_DATA\test\theoretical_1NoiseData1.npy'
        # 读取数据
        data_input = (np.load(data_path))[0:1280, 0:192]
        data_input = data_trans(data_input)
        data_input = data_input.unsqueeze(0)
        data_input = (Variable(data_input).to(device)).type(torch.float32)
        print(data_input.shape)
        # 模型预测概率
        y_pred = model(data_input)
        # pred，概率较大值对应的索引值，可看做预测结果，1表示行
        _, pred = torch.max(y_pred.data, 1)

        out = y_pred.squeeze(0)
        if start_epoch:
            imag_narry = (out.squeeze(0)).cpu().detach().numpy()
            Dao_num = imag_narry.shape[0]
            Points_num = imag_narry.shape[1]
            for i in range(Points_num):
                imag_narry[:, i] = normalization(imag_narry[:, i])
            x = np.arange(0, Dao_num, 1)
            y = np.arange(0, Points_num, 1)
            [x, y] = np.meshgrid(x, y)  # 转换成二维的矩阵坐标
            z = imag_narry[x, y]

            # plt.clf()  # 清除之前画的图
            ax.plot_surface(x, y, z, rstride=1, cstride=2, cmap='winter')
            plt.pause(10)
            plt.ioff()
            start_epoch = False

        # writer_array.add_scalar('epoch loss', epoch_loss, global_step=epoch)

        print('\n')

    time_end = time.time() - time_open
    print(time_end)