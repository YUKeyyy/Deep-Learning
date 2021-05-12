import torch
import torch.nn as nn
import torchvision

# model = torchvision.models.vgg16()
# del model.classifier
# print(model)
# print('\n')
# model = torchvision.models.resnet18()
# print(model)


# ranges 是用于方便获取和记录每个池化层得到的特征图
# 例如vgg16，需要(0, 5)的原因是为方便记录第一个pooling层得到的输出(详见下午、稳VGG定义)
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
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

#由cfg构建Vgg-net的卷积层和池化层
def Make_layers(cfg,batch_normal = False):
    layers = []
    #输入层的通道数
    in_channels = 3
    for v in cfg:
        if v =='M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_normal:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#m = Make_layers(cfg['vgg16'])

# class VGGnet(torchvision.models.VGG):
#     def __init__(self,pretrained = True,model='vgg16',requires_grad=True,remove_fc=True):
#         super(VGGnet,self).__init__(Make_layers(cfg[model]))
#         self.range = ranges[model]


class VGGnet(nn.Module):
    def __init__(self,pretrained = True,model='vgg16',requires_grad=True,remove_fc=True):
        super(VGGnet,self).__init__()
        self.layer = Make_layers(cfg[model])
        self.range = \
            [model]

    def forward(self,x):
        output = {}
        # 利用之前定义的ranges获取每个maxpooling层输出的特征图
        for idx, (begin, end) in enumerate(self.ranges):
            for layer in range(begin, end):
                x = self.features[layer](x)
                output["x%d"%(idx+1)] = x
        # output 为一个字典键x1d对应第一个maxpooling输出的特征图，x2...x5类推
        return output


class FCN8s(nn.Module):
    def __init__(self,pretrained_net,n_class):
        super(FCN8s,self).__init__()
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

    def forward(self,x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # maxpooling5的feature map (1/32)
        x4 = output['x4']  # maxpooling4的feature map (1/16)
        x3 = output['x3']  # maxpooling3的feature map (1/8)

        score = self.relu(self.conv6(x5))  # conv6  size不变 (1/32)
        score = self.relu(self.conv7(score))  # conv7  size不变 (1/32)
        score = self.relu(self.deconv1(x5))  # out_size = 2*in_size (1/16)
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))  # out_size = 2*in_size (1/8)
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  # out_size = 2*in_size (1/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # out_size = 2*in_size (1/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # out_size = 2*in_size (1)
        score = self.classifier(score)  # size不变，使输出的channel等于类别数



vgg_model = VGGnet()
print(FCN8s(vgg_model,7))