from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import os
from pycocotools.coco import COCO

import torchvision.datasets as dset

from demo import MonoCLIP

if __name__ == "__main__":
    
    version = "val2017"

    # COCO dataset
    json_path = "./dataset/annotations/instances_{}.json".format(version)  # 标注信息
    img_path = "./dataset/{}".format(version)
    
    save_path = "./outputs/{}".format(version)

    coco = COCO(json_path)
    ids = list(
        sorted(coco.imgs.keys())
    )  # 获取所有图片名称；ids 是所有图片id（图片名称后面的数字）的汇总排序

    print("number of images: {}".format(len(ids)))
    coco_classes = dict(
        [(v["id"], v["name"]) for k, v in coco.cats.items()]
    )  # create dict
    coco_cls = list(coco_classes.values())  # create list

    # Init Clip
    model = MonoCLIP(coco_cls)
    # remove resize and crop
    model.preprocess.transforms.pop(0)
    model.preprocess.transforms.pop(0)
    
    
    # process images
    for img_id in ids:
        path = coco.loadImgs(img_id)[0]["file_name"]  # 根据此图片的索引，获取图片名称
        img_data = Image.open(os.path.join(img_path, path)).convert(
            "RGB"
        )  # 根据图片路径打开图片
        image_name = path.split('.')[0]
        image = model.preprocess(img_data)

        h = image.shape[-2]
        w = image.shape[-1]

        # image_numpy = image.permute(1, 2, 0).cpu().numpy()
        # plt.imshow(image_numpy)

        input = image.to("cuda").unsqueeze(0)
        input_img_flip = torch.flip(input, [3])

        class_conf = model(input)
        
        # torch.save(class_conf.squeeze(0).detach(), os.path.join(save_path,image_name+".pth"))
        
        
        

    ##############################################################################

    for img_id in ids[:3]:  # 取验证集排序后的，前三张图像
        ann_ids = coco.getAnnIds(imgIds=img_id)  # 根据图像的索引，获取标注信息的索引
        targets = coco.loadAnns(ann_ids)  # 根据标注信息的索引，拿到标注信息
        path = coco.loadImgs(img_id)[0]["file_name"]  # 根据此图片的索引，获取图片名称
        img = Image.open(os.path.join(img_path, path)).convert(
            "RGB"
        )  # 根据图片路径打开图片
        draw = ImageDraw.Draw(img)

        # 开始绘制，将标注框画到图片上
        for target in targets:  # 将所有标注信息，绘制在图片上
            x, y, w, h = target["bbox"]
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
            draw.rectangle((x1, y1, x2, y2))  # 矩形框
            draw.text((x1, y1), coco_classes[target["category_id"]])  # 把类别写在左上角
        # 展示图片
        plt.imshow(img)
        plt.show()

    print("end")
