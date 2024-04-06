import torch
from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from open_clip import build_zero_shot_classifier
from collections import OrderedDict

seed_value = 42
torch.manual_seed(seed_value)

detection_templates = ["A photo of a {}"]
""" model_name: RN50, ViT-B-16 """


class MonoCLIP(nn.Module):
    def __init__(self, data_class: list, model_name="RN50", pre_trained="openai"):
        super(MonoCLIP, self).__init__()
        self.class_num = len(data_class)
        self.data_class = data_class

        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pre_trained
        )
        clip = clip.to("cuda")
        self.image_encoder = clip.visual.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        self.text_f = build_zero_shot_classifier(
            clip,
            tokenizer,
            self.data_class,
            detection_templates,
            device="cuda",
        )
        
        self.conv1 = nn.Conv2d(self.image_encoder.attnpool.v_proj.in_features,
                               self.image_encoder.attnpool.v_proj.out_features,
                               kernel_size=(1, 1)).to("cuda").to(torch.float32)
        self.conv2 = nn.Conv2d(self.image_encoder.attnpool.c_proj.in_features,
                               self.image_encoder.attnpool.c_proj.out_features,
                               kernel_size=(1, 1)).to("cuda").to(torch.float32)
        conv1_weight_shape = (*self.image_encoder.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.image_encoder.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.image_encoder.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.image_encoder.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.image_encoder.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.image_encoder.attnpool.c_proj.bias))

    def forward(self, x):
        with torch.no_grad():
            img_f = self.image_encoder(x)  # B, C, H, W
            img_f = self.conv1(img_f)
            img_f = self.conv2(img_f)
        
        B, C, H, W = img_f.shape
        img_f=img_f.reshape(-1,C,H*W).permute(0,2,1)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        # # new SCLIP with Vit
        # img_f /= img_f.norm(dim=-1, keepdim=True)
        # img_f = img_f[:, 1:]

        # patch_size = self.image_encoder.patch_size
        # w, h = x[0].shape[-2] // patch_size, x[0].shape[-1] // patch_size
        # # end

        # dataset class conf
        class_conf = img_f @ self.text_f
        class_conf = class_conf.permute(0, 2, 1).reshape(
            -1, self.class_num, H, W
        )  # B, K, H, W

        return class_conf
    
    
### vis func
def show_conf(class_conf):
    class_conf_np = class_conf.squeeze().cpu().detach().numpy()
    mask = np.argmax(class_conf_np, axis=0)
    plt.figure()
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.show()


def show_img(img):
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.imshow(img_np)
    plt.show()
