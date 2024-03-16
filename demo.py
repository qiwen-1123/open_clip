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

seed_value = 42
torch.manual_seed(seed_value)

depth_templates = ["This {} is {}"]
detection_templates = ["A photo of a {}"]

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

nusc_classes = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction vehicle",
    "bicycle",
    "motorcycle",
    "person",
    "traffic_cone",
    "barrier",
    "road surface",
    "traffic light",
    "street light",
    "traffic sign",
    "sidewalk",
    "building",
    "sky",
    "tree",
]

coco_cls=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"]

model_name = "convnext_large_d_320" # convnext_large_d_320, ViT-H-14-378-quickgelu, ViT-H-14
pre_trained = "laion2b_s29b_b131k_ft_soup"  # laion2b_s29b_b131k_ft_soup, dfn5b
tokenizer = open_clip.get_tokenizer(model_name)

def zeroshot_classifier(data_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for data_class in data_classes:
            texts = [
                template.format(data_class) for template in templates
            ]  # format with class
            texts=tokenizer(texts).cuda()
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
    def __init__(self, data_class:list,):
        super(MonoCLIP, self).__init__()
        self.class_num = len(data_class)
        self.data_class = data_class

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pre_trained
        )

        self.clip = self.clip.to("cuda")

        # self.text_f = zeroshot_classifier(
        #     self.data_class, detection_templates, self.clip
        # )  # init text feature
        
        self.text_f = build_zero_shot_classifier(self.clip, tokenizer,
                                                      self.data_class,
                                                      detection_templates,
                                                      device="cuda",)
        # last_text_f = torch.load("text_f.pth")
        # res = last_text_f - self.text_f
        # torch.save(self.text_f.detach(), "text_f.pth")

        # self.adapter = FCLayer(1024).to(self.clip.dtype)

    def forward(self, x):
        img_f = self.clip.encode_image(x)  # B, C, H, W
        h = img_f.shape[-2]
        w = img_f.shape[-1]
        img_f=img_f.reshape(-1,img_f.shape[-3],img_f.shape[-2]*img_f.shape[-1]).permute(0,2,1)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        # last = torch.load("img_f.pth")
        # res = (last - img_f).norm()
        # torch.save(img_f.detach(), "img_f.pth")

        # @: dot product of two vectors
        img_f = torch.nn.functional.interpolate(
            img_f, scale_factor=0.5
        )  # to match size

        # dataset class conf
        class_conf = 100 * img_f @ self.text_f
        class_conf = class_conf.permute(0, 2, 1).reshape(
            -1, self.class_num, h, w
        )  # B, K, H, W
        class_conf = F.softmax(class_conf, dim=1)

        return class_conf

if __name__ == "__main__":

    model = MonoCLIP(nusc_classes)
    image_ori = Image.open("000000217948.jpg")

    model.preprocess.transforms.pop(0)
    model.preprocess.transforms.pop(0)
    image = model.preprocess(image_ori)

    h=image.shape[-2]
    w=image.shape[-1]

    # image_numpy = image.permute(1, 2, 0).cpu().numpy()
    # plt.imshow(image_numpy)

    input = image.to("cuda").unsqueeze(0)
    input_img_flip = torch.flip(input, [3])

    class_conf = model(input)
    
    # last = torch.load("6832e717621341568c759151b5974512.pth")
    # res = (class_conf.squeeze(0)-last).norm()
    # torch.save(class_conf.squeeze(0).detach(), "0d0700a2284e477db876c3ee1d864668.pth")
    
    # add flip
    # class_conf_flip = model(input_img_flip)
    # class_conf_flip = torch.flip(class_conf_flip, [3])
    # class_conf = 0.5 * (class_conf + class_conf_flip)
    
    # interpolation
    # class_conf = nn.functional.interpolate(
    #     class_conf, size=(448, 448), mode="bilinear", align_corners=True
    # )
    
    # torch.save(class_conf.squeeze(0).detach(), "6832e717621341568c759151b5974512.pth")

    class_conf_np = class_conf.squeeze().cpu().detach().numpy()

    mask = np.argmax(class_conf_np, axis=0)

    plt.imshow(mask, cmap="jet", alpha=0.5)
    # plt.colorbar(label=nusc_classes,orientation='horizontal')

    # plt.imshow(image_new, alpha=0.5)
    

    plt.show()
    plt.close()
    print("done")