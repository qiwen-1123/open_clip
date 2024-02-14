import torch
from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


depth_templates = ["This {} is {}"]
obj_classes = ["object"]
depth_classes = [
    "giant",
    "extremely close",
    "close",
    "not in distance",
    "a little remote",
    "far",
    "unseen",
]
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1
clip_vis = "RN50"
model_name="RN50"
pre_trained="openai"


def zeroshot_classifier(depth_classes, obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [
                    template.format(obj, depth) for template in templates
                ]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


class FCLayer(nn.Module):
    def __init__(self, c_in=1024, reduction=4):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)
        
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pre_trained
        )


        self.clip = self.clip.to("cuda")
        self.text_f = zeroshot_classifier(
            depth_classes, obj_classes, depth_templates, self.clip
        )  # init text feature

        # self.adapter = FCLayer(1024).to(self.clip.dtype)

    def forward(self, x):
        img_f = self.clip.encode_image(x).permute(1, 0, 2)  # B, HW, C
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        # @: dot product of two vectors
        img_f = torch.nn.functional.interpolate(
            img_f, scale_factor=0.5
        )  # to match size

        depth_logits = (
            100.0 * img_f @ self.text_f
        )  # B, HW, K # img_f and text_f have both been normalized, so just use a inner product
        depth_logits = depth_logits.permute(0, 2, 1).reshape(
            -1, self.bins, 13, 17
        )  # B, K, H, W
        depth_logits /= temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor = torch.tensor(bin_list).to(depth.device)
        depth = depth * bin_tensor.reshape(1, self.bins).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        return depth




if __name__ == '__main__':

    Model = MonoCLIP()

    image = Image.open("demo.jpg")
    image = image.convert("RGB")
    image = image.resize((544, 416))

    image = np.asarray(image, dtype=np.float32) / 255.0

    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(image)

    input = tensor_image.to("cuda").unsqueeze(0)
    input_img_flip = torch.flip(input, [3])

    output_depth = Model(input)
    output_depth_flip = Model(input_img_flip)
    output_depth_flip = torch.flip(output_depth_flip, [3])
    output_depth = 0.5 * (output_depth + output_depth_flip)

    output_depth = nn.functional.interpolate(
        output_depth, size=[416, 544], mode="bilinear", align_corners=True
    )

    depth_np = output_depth.squeeze().cpu().detach().numpy()
    plt.imshow(depth_np, cmap="gray")
    plt.colorbar()  # 添加颜色条，用于显示深度值对应的颜色
    plt.show()


    print("done")
