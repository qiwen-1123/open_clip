from open_clip import TextEncoder
import open_clip
import torch
from PIL import Image
import numpy as np

from open_clip.openai import load_openai_model
from open_clip.coop.coop import PromptLearner, CustomCLIP, setup_cfg


from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import argparse

classnames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



### visual func
def vis_clip_conf(class_conf):
    import matplotlib.pyplot as plt
    class_conf_np = class_conf.squeeze().cpu().detach().numpy()
    mask = np.argmax(class_conf_np, axis=0)
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.show()
    
def show_img(img):
    import matplotlib.pyplot as plt
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()

def main(args):
    cfg = setup_cfg(args)
    model_name="ViT-B-16"
    pre_trained="openai"
    
    clip,_ , preprocess, = open_clip.create_model_and_transforms(
            model_name, pretrained=pre_trained
        )
    
    clip.dtype = torch.float32

    model = CustomCLIP(cfg, classnames, clip).to("cuda")
    
    image_ori = Image.open("./data/COCO_train2014_000000000825.jpg")
    image = preprocess(image_ori)
    
    input = image.to("cuda").unsqueeze(0)
    
    class_conf = model(input)

    
    print("done")


if __name__ == "__main__":
    
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = load_openai_model("ViT-B-16", device=device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
    

    
    print("done")