from datetime import datetime
import torch
from options import TrainOptions
from torch.utils.data import DataLoader
from dataset import *
from model import Model
import os
from torch.optim import Adam
from loss import intLoss
import kornia.filters as KF
import torch.nn.functional as F
from test import test
from transmodel import Model as Encoder_Model  # 预训练的编码器的model


# 权重初始化
def gaussian_weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            print('卷积权重初始化失败')
    elif isinstance(m, torch.nn.BatchNorm2d):
        try:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        except:
            print('BN初始化失败')

def dataLoader(opt, dataset):
    return DataLoader(
            dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=opt.n_workers,
            drop_last=True,
         )


# 加载训练好的vis编码器和ir编码器
def load_model(vis,ir):

    # 初始化预训练的编码器
    vis_model = Encoder_Model().cuda()
    ir_model = Encoder_Model().cuda()

    # 加载模型
    vis_model.load_state_dict(torch.load('visencoder/vis.model'))
    ir_model.load_state_dict(torch.load('irencoder/ir.model'))

    # 模型的模式为测试
    vis_model.eval()
    ir_model.eval()

    # 获得预训练的编码器的特征
    vis_fea = vis_model(vis)
    ir_fea = ir_model(ir)

    return vis_fea,ir_fea

def train(opt, dataset):
    # 加载数据
    dataloader = dataLoader(opt, dataset)
    # 训练轮数
    train_num = len(dataset)
    # model
    print('\n--- load model ---')
    print(f'------ 训练轮数为{train_num} --------')
    model = Model()
    # 初始化权重，放在cuda前面
    model.apply(gaussian_weights_init)
    # 设置网络优化器
    optimizer = Adam(model.parameters(), opt.lr)
    model.cuda()

    # 开始计时
    from datetime import datetime
    start_time = datetime.now()
    # count 用于打印数据，count%10==0打印损失函数，count%50=0保存模型一次
    count = 0
    batch_num = len(dataloader)
    # 训练的轮数， opt.epoch = 1
    for ep in range(opt.epoch):
        print('~~~Main 训练开始！~~~~')

        # 模型设置为train模式
        model.train()
        # 每个batchsize的训练
        for it, (img_ir, img_vi) in enumerate(dataloader):
            count += 1
            print(f'--第{ep}轮---{count} / {batch_num}----  ')

            # 优化器梯度清零
            optimizer.zero_grad()

            # 图片放入cuda
            if opt.gpu:
                img_vi = img_vi.cuda()
                img_ir = img_ir.cuda()

            # 获取预编码的特征图
            vis_fea,ir_fea = load_model(vis=img_vi,ir=img_ir) #  16,16,128,128

            # 生成的图片名命为 gen_image
            gen_iamge = model(vis=img_vi, ir=img_ir, vis_fea=vis_fea, ir_fea= ir_fea)

            # 梯度损失
            # 使用Sobel函数求导
            grad_ir = KF.spatial_gradient(img_ir, order=2).abs().sum(dim=[1,2])
            grad_vi = KF.spatial_gradient(img_vi, order=2).abs().sum(dim=[1,2])
            grad_fus = KF.spatial_gradient(gen_iamge, order=2).abs().sum(dim=[1,2])
            grad_joint = torch.max(grad_ir, grad_vi)
            # # 第二步：求 vis 和 ir 中用不上的梯度
            zeros = torch.zeros_like(grad_vi)
            ones = torch.ones_like(grad_vi)
            vis_dis = torch.where(grad_vi - grad_ir > 0, ones, zeros)
            ir_dis = 1 - vis_dis
            dis_vi = grad_vi * vis_dis  # [b,c,h,w]
            dis_ir = grad_ir * ir_dis  # [b,c,h,w]
            # 第三步：正相关是IF靠近联合梯度，负相关是IF原理用不上的梯度
            d_ap = torch.mean((grad_fus - grad_joint) ** 2)
            d_an_ir = torch.mean((grad_fus - dis_ir) ** 2)
            d_an_vi = torch.mean((grad_fus - dis_vi) ** 2)
            # 第四步：计算ContrastLoss
            loss_grad = d_ap / (d_an_vi + 1e-7) + d_ap / (d_an_ir + 1e-7)

            choose_ir = intLoss(img_ir)
            ones = torch.ones_like(img_ir)
            zeros = torch.zeros_like(img_ir)
            block_ir = torch.where(choose_ir > 0, ones, zeros)
            loss_intensity = F.l1_loss(gen_iamge*block_ir,img_ir*block_ir) + F.l1_loss(gen_iamge, img_vi)  
    
            loss_total = loss_grad + 5 * loss_intensity 

            # 总损失
            loss_total.backward()
            optimizer.step()

# ----------------------------------------------------------------------------------------------------------

            # 打印损失函数
            if count % 50 == 0:
                elapsed_time = datetime.now() - start_time
                print('loss_grad: %s, loss_ssim: %s, loss_total: %s,selapsed_time: %s' % (
                    loss_grad.item(), loss_intensity.item(),loss_total.item(), elapsed_time))

            if count % 200 == 0:
                # save model
                model.eval()
                model.cpu()
                save_model_filename = "Epoch_" + str(count) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join(opt.saveModelPath ,save_model_filename)
                torch.save(model.state_dict(), save_model_path)
                model.train()
                model.cuda()

            #  每个300轮验证一次结果
            if count % 400 == 0:
                # 因为计算机是并行的cpu的写入不能满足代码的调用速度，所以使用本轮前面的保存的一个model
                modelPath = "Epoch_" + str(count-200) + "_iters_" + str(count-200) + ".model"
                # modelPath = "Epoch_" + str(250) + "_iters_" + str(250) + ".model"
                test(modelPath)

    # 训练结束，保存最后一个模型
    model.eval()
    model.cpu()
    save_model_filename = "Final_epoch_" + str(count) + ".model"
    save_model_path = os.path.join(opt.saveModelPath, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)




if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = dataset(opt.traindata)
    train(opt,dataset)

