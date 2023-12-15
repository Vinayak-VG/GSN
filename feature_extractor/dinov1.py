import torch
import os
import argparse
import tqdm
import glob
import numpy as np
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lib.baselines import DINO, get_model
from sklearn.decomposition import PCA
import pickle as pk
from PIL import Image
import torchvision.transforms as transforms
import imageio

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--model", type=str, default="dino")
    parser.add_argument("--model_path", type=str, default="ckpts/dino_vitbase8_pretrain.pth")
    parser.add_argument("--imsize", default=(224, 224), nargs="+", type=int)
    parser.add_argument("--dir_images", default='../data/nerf_llff_data/flower/images_8', type=str)
    parser.add_argument("--main_folder", default='../../../../../media/Data1/vinayak/nerf_llff_data', type=str)
    # parser.add_argument("--dir_dst", type=str, default="data/di no")
    parser.add_argument("-f", default=0)
    args = parser.parse_args()
    return args

args = parse_args()
model = get_model(args.model, args.model_path, f"cuda:{args.gpu}")
image = torch.from_numpy(np.array(Image.open("/media/Data1/vinayak/nerf_llff_data/trex/images_4/DJI_20200223_163548_810.png"))).permute(2, 0, 1)
image = image/255.
image = image.float()

transform = transforms.Compose([
    # transforms.PILToTensor()
    # transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
image = transform(image).unsqueeze(0)
batch_feats = model.extract_features(image.float(), transform=False, upsample=False)
batch_feats = F.interpolate(batch_feats, scale_factor=8,
                     mode="bilinear", align_corners=True)

batch_feats = batch_feats[0, :3].permute(1, 2, 0).cpu()
batch_feats = (batch_feats - torch.min(batch_feats))/(torch.max(batch_feats) - torch.min(batch_feats))
imageio.imwrite(f'dinov2feat_vis_intermediate.png', (batch_feats.detach().cpu().numpy()*255).astype('uint8'))
        