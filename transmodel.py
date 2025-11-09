import torch.nn as nn
from layers import ConvLeakyRelu2d
import torch




class Encoder(nn.Module):
    def __init__(self,in_channel=1):
        super(Encoder, self).__init__()
        # 源图像拉成40个特征图
        self.conv1 = ConvLeakyRelu2d(in_channel, 16, norm='Batch', activation='ReLU')
        # 主干
        self.conv2 = ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU')
        self.conv3 = ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU')
        self.conv4 = ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU')
        self.conv5 = ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU')

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # return conv5
        # 返回编码到的5个特征
        return [conv1,conv2,conv3,conv4,conv5]

# 解码器的输入是 两个模块的连接
class Decoder(nn.Module):
    def __init__(self,in_channel= 16):
        super(Decoder, self).__init__()
        self.conv1 = ConvLeakyRelu2d(16,8, norm='Batch', activation='ReLU')
        self.conv2 = ConvLeakyRelu2d(8,1, norm='Batch',activation='Tanh')

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.en = Encoder()
        self.de = Decoder()

    def forward(self,image):
        en = self.en(image)
        # out = self.de(en) # 第一阶段训练需要解码，第二阶段融合无需解码
        # return out

        # 返回encoder的编码特征
        return en




