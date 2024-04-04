import torch
from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from open_clip import build_zero_shot_classifier, TextEncoder, PromptLearner

# from open_clip import TextEncoder, PromptLearner

seed_value = 42
torch.manual_seed(seed_value)

detection_templates = ["A photo of a {}"]
""" model_name: RN50, ViT-B-16 """


class MonoCLIP(nn.Module):
    def __init__(self, data_class: list, coop_cfg, model_name="ViT-B-16", pre_trained="openai"):
        super(MonoCLIP, self).__init__()
        self.class_num = len(data_class)
        self.data_class = data_class

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pre_trained
        )

        self.clip = self.clip.eval()
        self.clip.dtype=torch.float32
        
        # tokenizer = open_clip.get_tokenizer(model_name)
        # self.text_f = build_zero_shot_classifier(
        #     self.clip,
        #     tokenizer,
        #     self.data_class,
        #     detection_templates,
        #     device="cuda",
        # )
        
        self.prompt_learner = PromptLearner(coop_cfg, data_class, self.clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip)

        # self.adapter = FCLayer(1024).to(self.clip.dtype)

    def forward(self, x):
        with torch.no_grad():
            img_f = self.clip.encode_image(x)  # B, C, H, W
        # img_f=img_f.reshape(-1,img_f.shape[-3],img_f.shape[-2]*img_f.shape[-1]).permute(0,2,1)
        # img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        # new
        img_f /= img_f.norm(dim=-1, keepdim=True)
        img_f = img_f[:, 1:]

        patch_size = self.clip.visual.patch_size
        w, h = x[0].shape[-2] // patch_size, x[0].shape[-1] // patch_size
        # end

        # COOP text feature
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # @: dot product of two vectors
        # img_f = torch.nn.functional.interpolate(
        #     img_f, scale_factor=0.5
        # )  # to match size

        # dataset class conf
        # class_conf = img_f @ self.text_f
        class_conf = img_f @ text_features.t()
        class_conf = class_conf.permute(0, 2, 1).reshape(
            -1, self.class_num, h, w
        )  # B, K, H, W
        # class_conf = F.softmax(class_conf, dim=1)

        return class_conf
