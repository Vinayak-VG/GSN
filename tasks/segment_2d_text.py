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
# from region_grower_2d_trilateral import dev_region_grower_mask
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
import cv2
import torch
import pickle as pk
import open_clip
import matplotlib.pyplot as plt
import imageio

def patch_seg(clipfeats_pred, rgb_pred, gt_img, pca_file, thresh, path, global_step, gt_dino):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    model = model.cuda()

    with torch.autocast("cuda"):
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        text = tokenizer(["a plate"])
        text = text.cuda()
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        pca_reload = pk.load(open(pca_file,'rb'))
        text_features = pca_reload.transform(text_features.cpu().numpy())


    heat_map_clip = (clipfeats_pred.cuda().float() @ torch.from_numpy(text_features).cuda().float().T)[:, :, 0]
    heat_map_clip = (heat_map_clip - torch.min(heat_map_clip))/(torch.max(heat_map_clip) - torch.min(heat_map_clip))
    # mask = heat_map_clip > thresh
    # seg_rgb = mask.unsqueeze(-1).repeat(1, 1, 3).cuda() * rgb_pred.cuda()
    # bg_mask = torch.ones(seg_rgb.shape).cuda()
    # seg_rgb = bg_mask + seg_rgb
    # onegreater = seg_rgb > 1
    # onegreater = -1 * onegreater
    # seg_rgb = seg_rgb + onegreater
    cmap = plt.get_cmap('jet')
    heat_map_clip = cmap(heat_map_clip.cpu().numpy())
    heat_map_clip = np.delete(heat_map_clip, 3, 2)

    super_imposed_img = cv2.addWeighted(heat_map_clip.astype(np.float32), 0.6, gt_img.cpu().numpy(), 0.4, 0)
    # seg_rgb = (seg_rgb - torch.min(seg_rgb)) / (torch.max(seg_rgb) - torch.min(seg_rgb))
    # print(valid_mask.shape)
    # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
    folder_name = "seg_results_others/seg_cake_clip"
    os.makedirs(folder_name, exist_ok=True)
    # img_vis = seg_mask.cuda()
    # img_vis = torch.cat((rgb_pred.cuda(), seg_rgb.cuda(), torch.from_numpy(heat_map_clip).cuda()), dim = 1)
    # imageio.imwrite(f"chesstable_paperfig_strokes.png", img_vis.cpu().numpy()*255)
    imageio.imwrite(f"{folder_name}/stroke_seg_cake_{global_step}.png", super_imposed_img*255)
    imageio.imwrite(f"{folder_name}/cake_rgb_{global_step}.png", rgb_pred*255)
    # rgb_pred
    # quit()
    # os.makedirs("/home/sushanth/ZSGNT_AAAI/seg_results/chesstable_strokes_mask_vd", exist_ok=True)
    # imageio.imwrite(f"/home/sushanth/ZSGNT_AAAI/seg_results/chesstable_strokes_mask_vd/mask_{global_step}.png", valid_mask.cuda().repeat(1, 1, 3).cpu().numpy()*255)

# def patch_seg(dinofeats, rgb_pred, gt_img, pca_file, thresh, path, global_step, gt_dino):
# # def patch_seg(patch_file, rgb_file, pca_file, thresh):

#     # model = get_model('dino', 'feature_extractor/ckpts/dino_vitbase8_pretrain.pth', f"cuda:{0}")
#     with torch.autocast("cuda"):

#         # def save(features, filenames, output_path):
#         #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         #     features_dict = dict()
#         #     features = torch.Tensor(features.numpy()) # [N, C, H, W]
#         #     for idx, f in enumerate(filenames):
#         #         features_dict[os.path.basename(f)] = features
#         #     torch.save(features_dict, output_path)
    
#         # stroke = np.array(Image.open("/home/sushanth/ZSGNT_AAAI/strokes/stroke_chesstable.png"))
#         # stroke = stroke[:, :, 3] // 255

#         # stroke_index = np.where(stroke == 1)
       
#         # stroke_index_0 = torch.from_numpy(stroke_index[0])
#         # stroke_index_1 = torch.from_numpy(stroke_index[1])
       
#         # stroke_dinofeats = dinofeats[stroke_index_0, stroke_index_1]
#         # save(stroke_dinofeats, ["chesstable_stroke_feats.png"], 'strokes/chesstable_stroke_feats.pt')
#         # quit()
        

#         stroke_dinofeats = torch.load("/home/sushanth/ZSGNT_AAAI/strokes/chesstable_stroke_feats.pt")
#         stroke_dinofeats = stroke_dinofeats[list(stroke_dinofeats.keys())[0]].contiguous().view(-1, 64)

#         cluster_ids_x, cluster_centers = kmeans(
#             X=stroke_dinofeats, num_clusters=11, distance='euclidean', device=torch.device('cuda:0')
#         )
        
#         # print(cluster_centers.shape)
#         # save(cluster_centers, ["flower_patch_feats.png"], 'patches/flower_patch_feats.pt')
#         # quit()

#         dist = torch.zeros((dinofeats.shape[0], dinofeats.shape[1], 11))
#         for i in range(cluster_centers.shape[0]):
#             dist[:, :, i] = torch.mean((dinofeats.cuda() - cluster_centers[i].unsqueeze(0).unsqueeze(0).cuda())**2, dim = -1)
#         dist = torch.min(dist, dim=-1, keepdim=True)[0]
        
#         # patch = (patch - torch.min(patch)) / (torch.max(patch) - torch.min(patch))
#         # imageio.imwrite(f"patch.png", patch.cpu().numpy()*255)
#         # quit()
#         # rgb_pred = rgb_pred[]

#         # cluster_centers = torch.mean((patch_feats), dim = 0, keepdim=True)#.repeat(2, 1)
#         # dist = torch.mean((rgb_feats.cuda() - cluster_centers.unsqueeze(0).cuda())**2, dim = -1)
#         # dist = torch.zeros((rgb_feats.shape[0], rgb_feats.shape[1], 1),)
#         # for i in range(cluster_centers.shape[0]):
#         #     dist[:, :, i] = torch.mean((rgb_feats.cuda() - cluster_centers[i].unsqueeze(0).unsqueeze(0).cuda())**2, dim = -1)
#         # dist = torch.min(dist, dim=-1, keepdim=True)[0]

#         # H, W, _ = dinofeats.shape
#             # dinofeats = dinofeats.view(-1, 64)
#             # x = 3
#             # dist = torch.mean((dinofeats - cluster_centers_[x:x+1])**2, dim = -1, keepdim=True)
#             # dist, _ = kmeans_model.index.search(dinofeats.contiguous().view(-1, dim), 1)
#             # dist = torch.tensor(dist)

#         # rgb_feats_vis = rgb_feats[:, :, :3]
#         # rgb_feats_vis = (rgb_feats_vis - torch.min(rgb_feats_vis)) / (torch.max(rgb_feats_vis) - torch.min(rgb_feats_vis))
#         # imageio.imwrite(f"rgb_feats.png", rgb_feats_vis.cpu().numpy()*255)

#         dist_vis = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
#         # dist_hist = dist_vis.flatten().cpu().numpy()
#         # print(dist_hist.shape)
#         # fig, axs = plt.subplots(1, 1,
#         #                 figsize =(10, 7),
#         #                 tight_layout = True)
 
#         # axs.hist(dist_hist, bins = 20)
#         # plt.savefig(f"seg_tests/dist_hist{global_step}.png")
#         # print(dist_vis.shape)
#         # quit()
#         # dist_vis = dist.view(H, W, 1)
#         # imageio.imwrite(f"dist.png", dist_vis.cpu().numpy()*255)
#         valid_mask = (dist_vis < thresh).float()
        
#         # new_valid_mask = dev_region_grower_mask(valid_mask.permute(2, 0, 1).unsqueeze(0), dinofeats.permute(2, 0, 1).unsqueeze(0), rgb_pred.permute(2, 0, 1).unsqueeze(0))
#         # seg_mask = rgb_pred.cuda() * new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)
#         # img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3), seg_mask.cuda()), dim = 1)
#         # imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
#         # input()
#         # dinofeats_vis = dinofeats.clone()
#         # dinofeats_vis = (dinofeats_vis - torch.min(dinofeats_vis)) / (torch.max(dinofeats_vis) - torch.min(dinofeats_vis))

#         # for i in range(1000):
#         #     new_valid_mask = dev_region_grower_mask(new_valid_mask, dinofeats.permute(2, 0, 1).unsqueeze(0), rgb_pred.permute(2, 0, 1).unsqueeze(0))
#         #     seg_mask = rgb_pred.cuda() * new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)
            
#         #     img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3), seg_mask.cuda()), dim = 1)
#         #     imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
#         #     input()
#         # img_vis = torch.cat((valid_mask.cuda().repeat(1, 1, 3), new_valid_mask[0][0].unsqueeze(-1).cuda().repeat(1, 1, 3)), dim = 1)
#         # imageio.imwrite(f"seg_tests/region_growing_{global_step}.png", img_vis.cpu().numpy()*255)
#         # quit()

#         # valid_mask = valid_mask.view(H, W, 1)
#         # print(rgb_pred.shape, valid_mask.shape)

#         seg_mask = rgb_pred.cuda() * valid_mask.cuda().repeat(1, 1, 3)
#         bg_mask = torch.ones(seg_mask.shape).cuda()
#         seg_mask = bg_mask + seg_mask
#         onegreater = seg_mask > 1
#         onegreater = -1 * onegreater
#         seg_mask = seg_mask + onegreater
#         seg_mask = (seg_mask - torch.min(seg_mask)) / (torch.max(seg_mask) - torch.min(seg_mask))
#         # print(valid_mask.shape)
#         # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
#         folder_name = "seg_results/seg_tests_chesstable_strokes_vd"
#         os.makedirs(folder_name, exist_ok=True)
#         img_vis = seg_mask.cuda()
#         # imageio.imwrite(f"chesstable_paperfig_strokes.png", img_vis.cpu().numpy()*255)
#         imageio.imwrite(f"{folder_name}/stroke_seg_{global_step}.png", img_vis.cpu().numpy()*255)
#         os.makedirs("/home/sushanth/ZSGNT_AAAI/seg_results/chesstable_strokes_mask_vd", exist_ok=True)
#         imageio.imwrite(f"/home/sushanth/ZSGNT_AAAI/seg_results/chesstable_strokes_mask_vd/mask_{global_step}.png", valid_mask.cuda().repeat(1, 1, 3).cpu().numpy()*255)
#         # imageio.imwrite(f"patch_seg.png", seg_mask.cpu().numpy()*255)
# ###################################################################################################################################################

#         # cluster_centers = torch.mean((patch_feats), dim = 0, keepdim=True).repeat(2, 1)
#         # dist = torch.zeros((dinofeats.shape[0], dinofeats.shape[1], 2),)
#         # for i in range(cluster_centers.shape[0]):
#         #     dist[:, :, i] = torch.mean((dinofeats - cluster_centers[i].unsqueeze(0).unsqueeze(0))**2, dim = -1)
#         # dist = torch.min(dist, dim=-1, keepdim=True)[0]
#         # # H, W, _ = dinofeats.shape
#         #     # dinofeats = dinofeats.view(-1, 64)
#         #     # x = 3
#         #     # dist = torch.mean((dinofeats - cluster_centers_[x:x+1])**2, dim = -1, keepdim=True)
#         #     # dist, _ = kmeans_model.index.search(dinofeats.contiguous().view(-1, dim), 1)
#         #     # dist = torch.tensor(dist)
        
#         # dist_vis = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
#         # # dist_vis = dist.view(H, W, 1)
#         # imageio.imwrite(f"dist.png", dist_vis*255)
#         # valid_mask = (dist_vis < thresh).float()
#         # # valid_mask = valid_mask.view(H, W, 1)
#         # seg_mask = rgb_pred * valid_mask.repeat(1, 1, 3)
#         # # print(valid_mask.shape)
#         # # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
#         # imageio.imwrite(f"patch_seg.png", seg_mask*255)
#         # mask[:, :, valid_idx] = valid_mask.squeeze(-1)
        
# ###################################################################################################################################################

