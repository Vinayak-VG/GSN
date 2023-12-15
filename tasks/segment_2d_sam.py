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
from region_grower_2d_trilateral import dev_region_grower_mask
from segment_anything import sam_model_registry, SamPredictor
import cv2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = tfs.Compose([
        # tfs.Resize(256),
        # tfs.Resize(imsize),
        # tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=MEAN, std=STD)
    ])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
   
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

sam_checkpoint = "sam/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def patch_seg(dinofeats, rgb_pred, gt_img, pca_file, thresh, path, global_step, gt_dino):
# def patch_seg(patch_file, rgb_file, pca_file, thresh):
    # model = get_model('dino', 'feature_extractor/ckpts/dino_vitbase8_pretrain.pth', f"cuda:{0}")
    with torch.autocast("cuda"):

        # quit()
        
        # predictor = SamPredictor(sam)
        imageio.imwrite(f"rgb_sam.png", rgb_pred.cpu().numpy()*255)
        image = cv2.imread('rgb_sam.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        # plt.figure(figsize=(10,10))
        # plt.imsave("sam_image.png", image)
        # plt.axis('on')
        # plt.show()
        # input_box = np.array([425, 600, 700, 875])
        # if global_step == 0:
        # input_point = np.array([[250, 100], [300, 270], [150, 125]]) # Horns
        # input_point = np.array([[150, 150], [350, 200], [410, 300], [480, 250]]) # Trex
        # input_point = np.array([[400, 250], [350, 150]]) # Fortress
        # input_point = np.array([[350, 200], [425, 275], [250, 175]]) # Flower
        # elif global_step == 1:
        #     input_point = np.array([[150, 150], [220, 200]])
        # elif global_step == 2:
        #     input_point = np.array([[330, 190], [380, 125]])
        input_label = np.array([1, 1, 1])

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.savefig("sam/sam_point.png")
        # quit()
        # dinofeats = dinofeats.permute(2, 0, 1).unsqueeze(0)
        dinofeats = dinofeats.contiguous().view(-1, 64).cpu().numpy()
        # print(dinofeats.shape)
        pca_reload = pk.load(open("/home/sushanth/ZSGNT_AAAI/feature_extractor/pca/llff_sam_pcahorns.pkl",'rb'))
        mean_orig = np.load("/home/sushanth/ZSGNT_AAAI/feature_extractor/mean/mean_sam_horns.npy")
        # print(mean_orig.shape, dinofeats.shape)
        nComp = 64
        # dinofeats = F.interpolate(dinofeats, size=(48, 64))
        Xhat = np.dot(dinofeats[:,:nComp], pca_reload.components_[:nComp,:])
        Xhat = Xhat + mean_orig
        # print(Xhat.shape)
        dinofeats = torch.from_numpy(Xhat).cuda().view(378, 504, 256)
        # print(Xhat.shape)
        dinofeats = F.interpolate(dinofeats.permute(2, 0, 1).unsqueeze(0), size=(48, 64))
        # quit()

        predictor.features[:, :, :48] = dinofeats
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # masks, _, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_box[None, :],
        #     multimask_output=False,
        # )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(f"sam/sam64_horns_seg_gtsam_{global_step}_{i}.png", bbox_inches='tight', pad_inches=0)
            # plt.imsave(f"sam/sam64_trex_seg_gtsam_{global_step}_{i}.png", image)
        
        # quit()
        # if global_step == 2:
        #     quit()  
        # quit()
        # quit()
        # quit()
        # img = np.array(Image.open(patch_file))
        # img = img[:, :, :3]
        # i = Image.fromarray(img)
        # i.save(patch_file)  

        # patch = Image.open(patch_file).convert('RGB')
        # patch = transform(patch).unsqueeze(0)
        # patch = torch.nn.functional.interpolate(patch, ((patch.shape[2]//8)*8, (patch.shape[3]//8)*8)).float()
        
        # patch_feats = model.extract_features(patch, transform=False, upsample=False)

        # pca_reload = pk.load(open(pca_file,'rb'))
        # C, H_patch, W_patch = patch_feats[0].shape
        # patch_feats = patch_feats[0].permute(1, 2, 0).view(-1, C)
        # patch_feats = torch.from_numpy(pca_reload.transform(patch_feats.cpu().numpy()))#.view(H_patch, W_patch, 64).unsqueeze(0).permute(0, 3, 1, 2)

        # rgb = Image.open(rgb_file).convert('RGB')
        # rgb = transform(rgb).unsqueeze(0)
        # rgb = torch.nn.functional.interpolate(rgb, ((rgb.shape[2]//8)*8, (rgb.shape[3]//8)*8)).float()

        # rgb_feats = model.extract_features(rgb, transform=False, upsample=False)
        # rgb_feats = torch.nn.functional.interpolate(rgb_feats, (378, 504)).float()
        # # pca_reload = pk.load(open(pca_file,'rb'))
        # # C, H_rgb, W_rgb = rgb_feats[0].shape  
        # rgb_feats = rgb_feats[0].permute(1, 2, 0).view(-1, C)
        # rgb_feats = torch.from_numpy(pca_reload.transform(rgb_feats.cpu().numpy())).view(378, 504, 64)#.unsqueeze(0).permute(0, 3, 1, 2)

        # cluster_ids_x, cluster_centers = kmeans(
        #     X=patch_feats, num_clusters=11, distance='euclidean', device=torch.device('cuda:0')
        # )

        # patch = rgb_pred[100:260, 320:340] # Chess table
        # patch = rgb_pred[120:280, 255:297] # ColorFountain
        # patch = rgb_pred[120:220, 222:320] # Shoerack
        # # patch = rgb_pred[100:220, 1:50] # Stove
        # patch = rgb_pred[120:220, 320:340] # Flower
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
        # save(dinofeats[120:220, 222:320], ["shoerack_patch_feats.png"], 'patches/shoerack_patch_feats.pt')
        # quit()

        ##### Switch to Faiss Kmeans #####

        # patch_dinofeats = dinofeats[120:260, 255:297].contiguous().view(-1, 64)
        # print(patch_dinofeats.shape)
        # patch_dinofeats = torch.load("/home/sushanth/ZSGNT_AAAI/patches/shoerack_patch_feats.pt")
        # patch_dinofeats = patch_dinofeats[list(patch_dinofeats.keys())[0]].contiguous().view(-1, 64)
        # print(patch_dinofeats.shape)
        # # print(patch_dinofeats.shape)
        # # quit()
        # cluster_ids_x, cluster_centers = kmeans(
        #     X=patch_dinofeats, num_clusters=11, distance='euclidean', device=torch.device('cuda:0')
        # )
        
        # # print(cluster_centers.shape)
        # # save(cluster_centers, ["flower_patch_feats.png"], 'patches/flower_patch_feats.pt')
        # # quit()
        # dist = torch.zeros((dinofeats.shape[0], dinofeats.shape[1], 11))
        # for i in range(cluster_centers.shape[0]):
        #     dist[:, :, i] = torch.mean((dinofeats.cuda() - cluster_centers[i].unsqueeze(0).unsqueeze(0).cuda())**2, dim = -1)
        # dist = torch.min(dist, dim=-1, keepdim=True)[0]

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

        # dist_vis = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
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
        # valid_mask = (dist_vis < thresh).float()
        
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

        # seg_mask = rgb_pred.cuda() * valid_mask.cuda().repeat(1, 1, 3)
        # # print(valid_mask.shape)
        # # img_vis = torch.cat((valid_mask, dist_vis, dinofeats[:, :, 0:1]), dim = 1)
        # folder_name = "seg_tests_shoerack_vid"
        # os.makedirs(folder_name, exist_ok=True)
        # img_vis = torch.cat((rgb_pred.cuda(), seg_mask.cuda(), valid_mask.cuda().repeat(1, 1, 3)), dim = 1)
        # imageio.imwrite(f"{folder_name}/patch_seg_{global_step}.png", img_vis.cpu().numpy()*255)
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