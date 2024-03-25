import torch
# import faiss
from time import time
import pickle as pk
# import open_clip
import matplotlib.pyplot as plt
import imageio
import torchvision.transforms as tfs
from feature_extractor.lib.baselines import get_model
from PIL import Image
import numpy as np
import torch.nn.functional as F
from kmeans_pytorch.kmeans_pytorch import kmeans
import os
from matplotlib import pyplot as plt
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = tfs.Compose([
        # tfs.Resize(256),
        # tfs.Resize(imsize),
        # tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=MEAN, std=STD)
    ])

def patch_seg(dinofeats, rgb_pred, gt_img, pca_file, thresh, path, global_step, gt_dino):
# def patch_seg(patch_file, rgb_file, pca_file, thresh):

    # model = get_model('dino', 'feature_extractor/ckpts/dino_vitbase8_pretrain.pth', f"cuda:{0}")
    with torch.autocast("cuda"):

        # patch = rgb_pred[100:260, 320:338] # Chess table
        # patch = rgb_pred[100:260, 320:340] # Chess table
        # patch = rgb_pred[120:280, 255:297] # chesstable
        # patch = rgb_pred[120:250, 240:260] # Shoerack
        # patch = rgb_pred[110:190, 3:50] # Stove
        # patch = rgb_pred[120:220, 320:340] # Flower
        # patch = rgb_pred[130:280, 255:270] # Fern
        # patch1 = rgb_pred[100:260, 320:340] # Chess table
        # patch2 = rgb_pred[120:280, 255:297]
        # patch_feats = torch.cat((patch1.contiguous().view(-1, 64), patch2.contiguous().view(-1, 64)), dim = 0)
        # patch_vis = (patch - torch.min(patch)) / (torch.max(patch) - torch.min(patch))
        # imageio.imwrite(f"patch.png", patch_vis.cpu().numpy()*255)
        # quit()

        # def save(features, filenames, output_path):
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     features_dict = dict()
        #     features = torch.Tensor(features.numpy()) # [N, C, H, W]
        #     for idx, f in enumerate(filenames):
        #         features_dict[os.path.basename(f)] = features
        #     torch.save(features_dict, output_path)
        # save(dinofeats[120:250, 240:260], ["shoerack_patch_feats.png"], 'patches/shoerack_patch_feats_vd.pt')
        # quit()

        patch_dinofeats = torch.load("/home/sushanth/ZSGNT_AAAI/patches/shoerack_patch_feats.pt")
        patch_dinofeats = patch_dinofeats[list(patch_dinofeats.keys())[0]].contiguous().view(-1, 64)
        # print(patch_dinofeats.shape)
        # print(patch_dinofeats.shape)
        # quit()

        cluster_ids_x, cluster_centers = kmeans(
            X=patch_dinofeats, num_clusters=11, distance='euclidean', device=torch.device('cuda:0')
        )
        
        # print(cluster_centers.shape)
        # save(cluster_centers, ["flower_patch_feats.png"], 'patches/flower_patch_feats.pt')
        # quit()

        dist = torch.zeros((dinofeats.shape[0], dinofeats.shape[1], 11))
        for i in range(cluster_centers.shape[0]):
            dist[:, :, i] = torch.mean((dinofeats.cuda() - cluster_centers[i].unsqueeze(0).unsqueeze(0).cuda())**2, dim = -1)
        dist = torch.min(dist, dim=-1, keepdim=True)[0]
        
        # patch = (patch - torch.min(patch)) / (torch.max(patch) - torch.min(patch))
        # imageio.imwrite(f"patch.png", patch.cpu().numpy()*255)
        # quit()
        # rgb_pred = rgb_pred[]

        # cluster_centers = torch.mean((patch_feats), dim = 0, keepdim=True)#.repeat(2, 1)
        # dist = torch.mean((rgb_feats.cuda() - cluster_centers.unsqueeze(0).cuda())**2, dim = -1)
        # dist = torch.zeros((rgb_feats.shape[0], rgb_feats.shape[1], 1),)
        # for i in range(cluster_centers.shape[0]):
        #     dist[:, :, i] = torch.mean((rgb_feats.cuda() - cluster_centers[i].unsqueeze(0).unsqueeze(0).cuda())**2, dim = -1)
        # dist = torch.min(dist, dim=-1, keepdim=True)[0]

        # H, W, _ = dinofeats.shape
            # dinofeats = dinofeats.view(-1, 64)
            # x = 3
            # dist = torch.mean((dinofeats - cluster_centers_[x:x+1])**2, dim = -1, keepdim=True)
            # dist, _ = kmeans_model.index.search(dinofeats.contiguous().view(-1, dim), 1)
            # dist = torch.tensor(dist)

        # rgb_feats_vis = rgb_feats[:, :, :3]
        # rgb_feats_vis = (rgb_feats_vis - torch.min(rgb_feats_vis)) / (torch.max(rgb_feats_vis) - torch.min(rgb_feats_vis))
        # imageio.imwrite(f"rgb_feats.png", rgb_feats_vis.cpu().numpy()*255)

        dist_vis = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
        # dist_hist = dist_vis.flatten().cpu().numpy()
        # print(dist_hist.shape)
        # fig, axs = plt.subplots(1, 1,
        #                 figsize =(10, 7),
        #                 tight_layout = True)
 
        # axs.hist(dist_hist, bins = 20)
        # plt.savefig(f"seg_tests/dist_hist{global_step}.png")
        # print(dist_vis.shape)
        # quit()
        # dist_vis = dist.view(H, W, 1)
        # imageio.imwrite(f"dist.png", dist_vis.cpu().numpy()*255)
        valid_mask = (dist_vis < thresh).float()
        
        # new_valid_mask = dev_region_grower_mask(valid_mask.permute(2, 0, 1).unsqueeze(0), dinofeats.permute(2, 0, 1).unsqueeze(0), rgb_pred.permute(2, 0, 1).unsqueeze(0))
        # seg_mask = rgb_pred.cuda() * new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)
        # img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3), seg_mask.cuda()), dim = 1)
        # imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
        # input()
        # dinofeats_vis = dinofeats.clone()
        # dinofeats_vis = (dinofeats_vis - torch.min(dinofeats_vis)) / (torch.max(dinofeats_vis) - torch.min(dinofeats_vis))

        # for i in range(1000):
        #     new_valid_mask = dev_region_grower_mask(new_valid_mask, dinofeats.permute(2, 0, 1).unsqueeze(0), rgb_pred.permute(2, 0, 1).unsqueeze(0))
        #     seg_mask = rgb_pred.cuda() * new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)
            
        #     img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3), seg_mask.cuda()), dim = 1)
        #     imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
        #     input()
        # img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)), dim = 1)
        # imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
        # quit()

        # valid_mask = valid_mask.view(H, W, 1)
        # print(rgb_pred.shape, valid_mask.shape)
        rgb_pred = torch.clamp(rgb_pred, 0, 1)
        seg_mask = rgb_pred.cuda() * valid_mask.cuda().repeat(1, 1, 3)
        # seg_mask[~valid_mask.bool().cuda().repeat(1, 1, 3)] = 1.0
        seg_mask = seg_mask + (1 - valid_mask.cuda().repeat(1, 1, 3))
        # seg_mask = (seg_mask - torch.min(seg_mask)) / (torch.max(seg_mask) - torch.min(seg_mask))
        # bg_mask = torch.ones(seg_mask.shape).cuda()
        # seg_mask = bg_mask + seg_mask
        # onegreater = seg_mask > 1
        # onegreater = -1 * onegreater
        # seg_mask = seg_mask + onegreater
        # print(valid_mask.shape)
        # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
        folder_name = "seg_results/seg_tests_shoerack_patches_vid"
        os.makedirs(folder_name, exist_ok=True)
        img_vis = (seg_mask.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(f"{folder_name}/seg_patch_{global_step}.png", img_vis)
        # os.makedirs(f"/home/sushanth/ZSGNT_AAAI/seg_results/shoerack_patch_mask_vd", exist_ok=True)
        # imageio.imwrite(f"/home/sushanth/ZSGNT_AAAI/seg_results/shoerack_patch_mask_vd/mask_{global_step}.png", valid_mask.cuda().repeat(1, 1, 3).cpu().numpy()*255)
      # imageio.imwrite(f"patch_seg.png", seg_mask.cpu().numpy()*255)
###################################################################################################################################################

        # cluster_centers = torch.mean((patch_feats), dim = 0, keepdim=True).repeat(2, 1)
        # dist = torch.zeros((dinofeats.shape[0], dinofeats.shape[1], 2),)
        # for i in range(cluster_centers.shape[0]):
        #     dist[:, :, i] = torch.mean((dinofeats - cluster_centers[i].unsqueeze(0).unsqueeze(0))**2, dim = -1)
        # dist = torch.min(dist, dim=-1, keepdim=True)[0]
        # # H, W, _ = dinofeats.shape
        #     # dinofeats = dinofeats.view(-1, 64)
        #     # x = 3
        #     # dist = torch.mean((dinofeats - cluster_centers_[x:x+1])**2, dim = -1, keepdim=True)
        #     # dist, _ = kmeans_model.index.search(dinofeats.contiguous().view(-1, dim), 1)
        #     # dist = torch.tensor(dist)
        
        # dist_vis = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
        # # dist_vis = dist.view(H, W, 1)
        # imageio.imwrite(f"dist.png", dist_vis*255)
        # valid_mask = (dist_vis < thresh).float()
        # # valid_mask = valid_mask.view(H, W, 1)
        # seg_mask = rgb_pred * valid_mask.repeat(1, 1, 3)
        # # print(valid_mask.shape)
        # # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
        # imageio.imwrite(f"patch_seg.png", seg_mask*255)
        # mask[:, :, valid_idx] = valid_mask.squeeze(-1)
        
###################################################################################################################################################

# patch_seg("/home/rg/vinayak/ZSGNT/patches/stove_plate.png", "/media/Data1/vinayak/seg_data/data4_stove/images_4/image000.png", "/home/rg/vinayak/ZSGNT/feature_extractor/pca/llff_dino_pcadata4_stove.pkl", 0.2)