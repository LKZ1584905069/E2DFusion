from datetime import datetime
import torch
from options import TrainOptions
from torch.utils.data import DataLoader
from dataset import *
from transmodel import Model
import os
from torch.optim import Adam,RMSprop
from loss import final_ssim,SSIM,MySSIM
import kornia.filters as KF
import torch.nn.functional as F
from transtest import test

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
            drop_last=False,
         )


def train(opt, dataset):
    # 加载数据
    dataloader = dataLoader(opt, dataset)
    # 训练轮数
    train_num = len(dataset)
    # model
    print('\n--- load model ---')
    print(f'------ 训练轮数为{train_num} --------')
    model = Model()

    model.apply(gaussian_weights_init)

    model.cuda()

    # 开始计时
    from datetime import datetime
    start_time = datetime.now()
    # count 用于打印数据，count%10==0打印损失函数，count%50=0保存模型一次
    count = 0
    batch_num = len(dataloader)
    # 训练的轮数， opt.epoch = 1
    for ep in range(opt.epoch):
        print('~~~Main_GAN 训练开始！~~~~')

        # 模型设置为train模式
        model.train()

        # 每个batchsize的训练
        for it, (img_ir, img_vi) in enumerate(dataloader):
            count += 1
            print(f'--第{ep}轮---{count} / {batch_num}----  ')

            # 设置网络优化器
            optimizer_G = Adam(model.parameters(), opt.lr)


            # 优化器梯度清零
            optimizer_G.zero_grad()

            # 图片放入cuda
            if opt.gpu:
                img_vi = img_vi.cuda()
                img_ir = img_ir.cuda()

            source = img_ir
            # 生成的图片名命为 gen_image
            gen_iamge = model(image=source)

            ssim_loss = SSIM(window_size=11)
            loss_ssim = 1 - ssim_loss(gen_iamge,source)
            loss_intensity = F.l1_loss(gen_iamge,source)

            loss_total = 10 * loss_ssim + loss_intensity

            loss_total.backward()

            optimizer_G.step()

# ----------------------------------------------------------------------------------------------------------

            # 打印损失函数
            if count % 50 == 0:
                elapsed_time = datetime.now() - start_time
                print('loss_intensity: %s, loss_ssim: %s, loss_total: %s,selapsed_time: %s' % (
                     loss_intensity.item(),loss_ssim.item(),loss_total.item(), elapsed_time))

            if count % 500 == 0:
                # save model
                model.eval()
                model.cpu()
                save_model_filename = "Epoch_" + str(count) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join('./visencoder',save_model_filename)
                torch.save(model.state_dict(), save_model_path)
                model.train()
                model.cuda()

            #  每个300轮验证一次结果
            if count % 1000 == 0:
                # 因为计算机是并行的cpu的写入不能满足代码的调用速度，所以使用本轮前面的保存的一个model
                modelPath = "Epoch_" + str(count-500) + "_iters_" + str(count-500) + ".model"
                test(modelPath)

    # 训练结束，保存最后一个模型
    model.eval()
    model.cpu()
    save_model_filename = "Final_epoch_" + str(count) + ".model"
    # args.save_model_dir = models
    save_model_path = os.path.join('./visencoder', save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)




if __name__ == '__main__':
    opt = TrainOptions().parse()

    is_train = True

    if is_train:
        # 数据加载
        dataset = dataset(opt.traindata)
        train(opt,dataset)

