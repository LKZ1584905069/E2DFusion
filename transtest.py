 # test phase

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import utils
import numpy as np
import time
from transmodel import Model
from glob import glob
from PIL import Image
import torchvision.transforms.functional as TF
import cv2


def load_model(path):
    fuse_net = Model()
    fuse_net.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))
    fuse_net.eval()
    return fuse_net.cuda()


def run_demo(model, vi_path, ir_path, output_path_root, index):
    source_image = Image.open(ir_path)
    source_image = TF.to_tensor(source_image).unsqueeze(0).cuda()

    img_fusion = model(source_image)
    file_name =  ir_path.split('/')[-1]
    output_path = output_path_root + file_name

    img_fusion = np.array(img_fusion.squeeze().cpu()*255)
    img = img_fusion.astype('uint8')

    cv2.imwrite(output_path, img)
    print(output_path)


def test(model_path = None):
    output_path = 'visresult/'


    # 读取每个文件夹里面的图片名称
    train_ir_data_names = glob('./testimgs/ir/*')  # 实际训练使用
    train_vi_data_names = glob('./testimgs/vi/*')  # 实际训练使用

    train_vi_data_names.sort()
    train_ir_data_names.sort()
    print(train_vi_data_names)
    # model_path = "./irencoder/" + model_path
    model_path = "irencoder/ir.model"


    with torch.no_grad():
        # 加载模型
        model_test = load_model(model_path)
        # 每张图片生成
        for i in range(len(train_ir_data_names)):
            start = time.time()
            run_demo(model_test, train_vi_data_names[i], train_ir_data_names[i], output_path, i)
            end = time.time()
            print('time:', end - start, 'S')
    print('Done......')


if __name__ == "__main__":
    test()