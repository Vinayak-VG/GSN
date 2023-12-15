import torch
import os
import argparse
import tqdm
import glob
import numpy as np
import torchvision.transforms as tfs
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
import matplotlib.pyplot as plt

from lib.baselines import DINO, get_model
from sklearn.decomposition import PCA
import pickle as pk

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
    parser.add_argument("--main_folder", default='../../../../../media/Data1/vinayak/real_iconic_noface/', type=str)
    # parser.add_argument("--dir_dst", type=str, default="data/di no")
    parser.add_argument("-f", default=0)
    args = parser.parse_args()
    return args


def get_dataloader(args):
    transform = get_default_transform(args.imsize)

    # Image dataset
    dataset = ImageFolderNoLabels(args.dir_images, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dataloader


def save(features, filenames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_dict = dict()
    features = torch.Tensor(features.numpy()) # [N, C, H, W]
    for idx, f in enumerate(filenames):
        features_dict[os.path.basename(f)] = features[idx, ...]
    torch.save(features_dict, output_path)


def get_default_transform(imsize):
    transform = tfs.Compose([
        # tfs.Resize(256),
        tfs.Resize(imsize),
        # tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=MEAN, std=STD)
    ])
    return transform


class ImageFolderNoLabels(VisionDataset):
    def __init__(self, root, transform, loader=default_loader, is_valid_file=None):
        root = os.path.abspath(root)
        super().__init__(root, transform)
        self.loader = loader
        samples = self.parse_dir(root, IMG_EXTENSIONS if is_valid_file is None else None)
        self.samples = samples


    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, self.samples[index]

    def __len__(self):
        return len(self.samples)

    def parse_dir(self, dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not os.path.isdir(dir):
            raise IOError(f"{dir} is not a directory.")
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        # parse
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images.append(path)
        return images


def main(args=None):
    scenes = sorted(os.listdir(os.path.join(args.main_folder)))
    print(scenes)
    # model = get_model(args.model, args.model_path, f"cuda:{args.gpu}")
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = DINO()
    for scene in tqdm.tqdm(scenes):
        args.dir_images = os.path.join(args.main_folder, scene, 'images_4')
        im = plt.imread(glob.glob(args.dir_images + '/*')[0])
        args.imsize = tuple([(im.shape[0]//14)*14, (im.shape[1]//14)*14])
        dataloader = get_dataloader(args)
        all_filenames = []
        all_features = []
        for batch, filenames in dataloader:

            batch_feats = model.extract_features(batch, transform=False, upsample=False)
            all_filenames.extend(filenames)
            all_features.append(batch_feats.detach().cpu())

        all_features = torch.cat(all_features, 0)
        pca = PCA(n_components=args.pca)
        N, C, H, W = all_features.shape
        all_features = all_features.permute(0, 2, 3, 1).view(-1, C).numpy()
        X = pca.fit_transform(all_features)
        X = torch.Tensor(X).view(N, H, W, args.pca).permute(0, 3, 1, 2)
        os.makedirs(os.path.join(args.main_folder, scene, 'DiNOFeats_v2'), exist_ok=True)
        # pk.dump(pca, open(f"pca/llff_dino_pca{scene}.pkl","wb"))
        
        for idx in range(len(all_filenames)):
            output_path_pca = os.path.join(args.main_folder, scene, 'DiNOFeats_v2', os.path.basename(all_filenames[idx].split('.')[0]) + ".pt")
            # print(f'Saving features to {output_path_pca}')
            save(X[idx].unsqueeze(0), all_filenames[idx:idx+1], output_path_pca)

if __name__ == "__main__":
    args = parse_args()
    main(args)
# 