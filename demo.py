import torch
from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from open_clip import build_zero_shot_classifier, MonoCLIP

from open_clip import setup_cfg, generate_args # COOP part

seed_value = 42
torch.manual_seed(seed_value)

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

# coco_cls=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"]
coco_cls=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def show_img(img):
    import matplotlib.pyplot as plt
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()

if __name__ == "__main__":
    args = generate_args()
    cfg = setup_cfg(args)

    model = MonoCLIP(data_class=coco_cls, coop_cfg=cfg).to("cuda")
    image_ori = Image.open("./data/000000252219.jpg")

    # model.preprocess.transforms.pop(0)
    # model.preprocess.transforms.pop(0)
    image = model.preprocess(image_ori)

    h=image.shape[-2]
    w=image.shape[-1]

    input = image.to("cuda").unsqueeze(0)
    input_img_flip = torch.flip(input, [3])

    class_conf = model(input)
    
    # interpolation
    # class_conf = nn.functional.interpolate(
    #     class_conf, size=(448, 448), mode="bilinear", align_corners=True
    # )
    
    class_conf_np = class_conf.squeeze().cpu().detach().numpy()

    mask = np.argmax(class_conf_np, axis=0)

    plt.imshow(mask, cmap="jet", alpha=0.5)

    plt.show()
    plt.close()
    print("done")