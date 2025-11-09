import torch.nn as nn
from layers import ConvLeakyRelu2d
import torch


# 空间注意力机制采用max（VIS空间，IR空间）* 混合特征
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(16, 16 // 4, 3, padding=1)
        self.conv2 = nn.Conv2d(16 // 4, 1, 3, padding=1)
        self.ReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        return self.sigmoid(x)


# 通道注意力机制采用（混合 - IR）作为VIS的通道注意力特征 * VIS，同理 IR 也是
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=16, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.shared_MLP2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP1(self.avg_pool(x))
        max_out = self.shared_MLP2(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


'''
    ChannelAttention为K或者Q
    SpatialAttention为K或者Q
    特征图本身为V
'''


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        # 四类卷积
        self.Vconv1 = ConvLeakyRelu2d(16, 16, kernel_size=1, padding=0, norm='BN', activation='ReLU')
        self.Vconv3 = ConvLeakyRelu2d(16, 16, kernel_size=3, padding=1, norm='BN', activation='ReLU')
        self.Vconv5 = ConvLeakyRelu2d(16, 16, kernel_size=5, padding=2, norm='BN', activation='ReLU')
        self.Vconv7 = ConvLeakyRelu2d(16, 16, kernel_size=7, padding=3, norm='BN', activation='ReLU')

        self.Iconv1 = ConvLeakyRelu2d(16, 16, kernel_size=1, padding=0, norm='BN', activation='ReLU')
        self.Iconv3 = ConvLeakyRelu2d(16, 16, kernel_size=3, padding=1, norm='BN', activation='ReLU')
        self.Iconv5 = ConvLeakyRelu2d(16, 16, kernel_size=5, padding=2, norm='BN', activation='ReLU')
        self.Iconv7 = ConvLeakyRelu2d(16, 16, kernel_size=7, padding=3, norm='BN', activation='ReLU')

        # 四类空间和通道注意力机制
        # VCn 代表是VIS的n类卷积的Channel，VSn 是 Spatial
        # ICn 是IR的
        self.VC1 = ChannelAttention()
        self.VS1 = SpatialAttention()
        self.VC3 = ChannelAttention()
        self.VS3 = SpatialAttention()
        self.VC5 = ChannelAttention()
        self.VS5 = SpatialAttention()
        self.VC7 = ChannelAttention()
        self.VS7 = SpatialAttention()

        self.IC1 = ChannelAttention()
        self.IS1 = SpatialAttention()
        self.IC3 = ChannelAttention()
        self.IS3 = SpatialAttention()
        self.IC5 = ChannelAttention()
        self.IS5 = SpatialAttention()
        self.IC7 = ChannelAttention()
        self.IS7 = SpatialAttention()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, vis, ir):
        # 原始特征图经过不同的卷积分为四类
        V1 = self.Vconv1(vis)
        V3 = self.Vconv3(vis)
        V5 = self.Vconv5(vis)
        V7 = self.Vconv7(vis)

        I1 = self.Iconv1(ir)
        I3 = self.Iconv3(ir)
        I5 = self.Iconv5(ir)
        I7 = self.Iconv7(ir)

        # VIS 和 IR 的通道和空间注意力机制结果
        VC1 = self.VC1(V1 - I1)
        VC3 = self.VC1(V3 - I3)
        VC5 = self.VC1(V5 - I5)
        VC7 = self.VC1(V7 - I7)
        VS1 = self.VS1(V1 - I1)
        VS3 = self.VS1(V3 - I3)
        VS5 = self.VS1(V5 - I5)
        VS7 = self.VS1(V7 - I7)

        IC1 = self.IC1(I1 - V1)
        IC3 = self.IC1(I3 - V3)
        IC5 = self.IC1(I5 - V5)
        IC7 = self.IC1(I7 - V7)
        IS1 = self.IS1(I1 - V1)
        IS3 = self.IS1(I3 - V3)
        IS5 = self.IS1(I5 - V5)
        IS7 = self.IS1(I7 - V7)

        # VIS 的 K * Q
        V1KQ1 = VS1 * VC1
        V1KQ3 = VS1 * VC3
        V1KQ5 = VS1 * VC5
        V1KQ7 = VS1 * VC7

        V3KQ1 = VS3 * VC1
        V3KQ3 = VS3 * VC3
        V3KQ5 = VS3 * VC5
        V3KQ7 = VS3 * VC7

        V5KQ1 = VS5 * VC1
        V5KQ3 = VS5 * VC3
        V5KQ5 = VS5 * VC5
        V5KQ7 = VS5 * VC7

        V7KQ1 = VS7 * VC1
        V7KQ3 = VS7 * VC3
        V7KQ5 = VS7 * VC5
        V7KQ7 = VS7 * VC7

        # IR 的 Q * K
        I1KQ1 = self.softmax(IS1 * IC1)
        I1KQ3 = self.softmax(IS1 * IC3)
        I1KQ5 = self.softmax(IS1 * IC5)
        I1KQ7 = self.softmax(IS1 * IC7)

        I3KQ1 = self.softmax(IS3 * IC1)
        I3KQ3 = self.softmax(IS3 * IC3)
        I3KQ5 = self.softmax(IS3 * IC5)
        I3KQ7 = self.softmax(IS3 * IC7)

        I5KQ1 = self.softmax(IS5 * IC1)
        I5KQ3 = self.softmax(IS5 * IC3)
        I5KQ5 = self.softmax(IS5 * IC5)
        I5KQ7 = self.softmax(IS5 * IC7)

        I7KQ1 = self.softmax(IS7 * IC1)
        I7KQ3 = self.softmax(IS7 * IC3)
        I7KQ5 = self.softmax(IS7 * IC5)
        I7KQ7 = self.softmax(IS7 * IC7)

        # VIS 的 KQ * V
        VV1 = V1KQ1 * V1 + V1KQ3 * V3 + V1KQ5 * V5 + V1KQ7 * V7
        VV3 = V3KQ1 * V1 + V3KQ3 * V3 + V3KQ5 * V5 + V3KQ7 * V7
        VV5 = V5KQ1 * V1 + V5KQ3 * V3 + V5KQ5 * V5 + V5KQ7 * V7
        VV7 = V7KQ1 * V1 + V7KQ3 * V3 + V7KQ5 * V5 + V7KQ7 * V7

        IV1 = I1KQ1 * I1 + I1KQ3 * I3 + I1KQ5 * I5 + I1KQ7 * I7
        IV3 = I3KQ1 * I1 + I3KQ3 * I3 + I3KQ5 * I5 + I3KQ7 * I7
        IV5 = I5KQ1 * I1 + I5KQ3 * I3 + I5KQ5 * I5 + I5KQ7 * I7
        IV7 = I7KQ1 * I1 + I7KQ3 * I3 + I7KQ5 * I5 + I7KQ7 * I7

        # return (VV1+VV3+VV5+VV7),(IV1+IV3+IV5+IV7)
        return torch.cat((VV1 + IV1, VV3 + IV3, VV5 + IV5, VV7 + IV7), 1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.de1 = ConvLeakyRelu2d(64, 32, norm='Batch', activation='LReLU')
        self.de2 = ConvLeakyRelu2d(32, 16, norm='Batch', activation='LReLU')
        self.de3 = ConvLeakyRelu2d(16, 1, activation='Tanh')

    def forward(self, x):

        de1 = self.de1(x)
        de2 = self.de2(de1)
        out = self.de3(de2)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vis1 = nn.Sequential(ConvLeakyRelu2d(1, 16, norm='Batch', activation='ReLU'),
                                   ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU'))
        self.vis2 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                   ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.vis3 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                   ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.vis4 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                   ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.vis5 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                   ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))

        self.ir1 = nn.Sequential(ConvLeakyRelu2d(1, 16, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(16, 16, norm='Batch', activation='ReLU'))
        self.ir2 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.ir3 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.ir4 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))
        self.ir5 = nn.Sequential(ConvLeakyRelu2d(32, 32, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(32, 16, norm='Batch', activation='ReLU'))

        # 注意力
        self.Attention = Attention()

        # 解码
        self.de = Decoder()

    def forward(self, vis, ir, vis_fea, ir_fea):
        # vis分支
        vis1 = self.vis1(vis)
        vis2 = self.vis2(torch.cat((vis1, vis_fea[0]), 1))
        vis3 = self.vis3(torch.cat((vis2, vis_fea[1]), 1))
        vis4 = self.vis4(torch.cat((vis3, vis_fea[2]), 1))
        vis5 = self.vis5(torch.cat((vis4, vis_fea[3]), 1))

        # ir分支
        ir1 = self.ir1(ir)
        ir2 = self.ir2(torch.cat((ir1, ir_fea[3]), 1))
        ir3 = self.ir3(torch.cat((ir2, ir_fea[2]), 1))
        ir4 = self.ir4(torch.cat((ir3, ir_fea[1]), 1))
        ir5 = self.ir5(torch.cat((ir4, ir_fea[0]), 1))

        # attention
        att = self.Attention(vis=vis5, ir=ir5)

        # 解码
        out = self.de(att)

        return out





