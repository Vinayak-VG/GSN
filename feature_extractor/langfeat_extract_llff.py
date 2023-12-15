import torch
import open_clip
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from sklearn.decomposition import PCA
import tqdm
import os
import pickle as pk

transform = T.ToPILImage()
size = 96 # patch size
stride = 48 # patch stride.
src = "/mnt/data/sushanth/vinayak/data/shiny"
scenes = sorted(os.listdir(os.path.join(src)))
model_names = ['ViT-B-16']
dataset_names = ['laion2b_s34b_b88k']

# tokenizer = open_clip.get_tokenizer('ViT-B-16')
def save(features, filenames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_dict = dict()
    features = torch.Tensor(features.cpu().numpy()) # [N, C, H, W]
    for idx, f in enumerate(filenames):
        features_dict[os.path.basename(f)] = features[idx, ...]
    torch.save(features_dict, output_path)


for scene in tqdm.tqdm(range(len(scenes))):
    # print(f"Current Scene: {scenes[scene]}")
    images = sorted(os.listdir(os.path.join(src, scenes[scene], 'images_4')))
    os.makedirs(os.path.join(src, scenes[scene], 'CLIPFeats'), exist_ok=True)
    features = torch.zeros((len(images), 14, 20, len(model_names), 512)).cuda()
    for model_num in range(len(model_names)):
        model, _, preprocess = open_clip.create_model_and_transforms(model_names[model_num], pretrained=dataset_names[model_num])
        model = model.cuda()
        for image in range(len(images)):
            image_path = os.path.join(src, scenes[scene], 'images_4', images[image])
            x = torch.from_numpy(np.array(Image.open(image_path))[:, :, :3]).permute(2, 0, 1).unsqueeze(0).float()
            patches = x.unfold(2, size, stride).unfold(3, size, stride)
            patches = np.array(patches).astype(np.uint8)
            with torch.autocast("cuda"):
                patches = torch.from_numpy(patches)
                for i in range(patches.shape[2]):
                    for j in range(patches.shape[3]):
                        clipimage = preprocess(transform(patches[0, :, i, j])).unsqueeze(0).cuda()
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            image_features = model.encode_image(clipimage)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            features[image][i][j][model_num] = image_features[0]

    features = features.view(features.shape[0], features.shape[1], features.shape[2], -1)
    pca = PCA(n_components=64)
    I, H, W, C = features.shape
    features = features.view(-1, C).cpu().numpy()
    features = pca.fit_transform(features)
    features = torch.Tensor(features).view(I, H, W, 64).permute(0, 3, 1, 2)
    pk.dump(pca, open(f"pca/llff_clip_pca{scenes[scene]}.pkl","wb"))

    for image in range(len(images)):
        output_path_pca = os.path.join(src, scenes[scene], 'CLIPFeats',  images[image].split('.')[0] + ".pt")
        save(features[image].unsqueeze(0), [images[image].split('.')[0]], output_path_pca)

# features = torch.Tensor(features).view(N, I, H, W, 64)
# for scene in tqdm.tqdm(range(len(scenes))):
#     images = sorted(os.listdir(os.path.join(src, 'Rectified', scenes[scene])))
#     os.makedirs(os.path.join(src, 'CLIPFeats'), exist_ok=True)
#     for image in range(len(images)):
#         output_path_pca = os.path.join(src, 'CLIPFeats', scenes[scene], images[image].split('.')[0] + ".pt")
#         save(features[scene][image].unsqueeze(0), [images[image].split('.')[0]], output_path_pca)

# pca_reload = pk.load(open("pca.pkl",'rb'))
# result_new = pca_reload .transform(X)
