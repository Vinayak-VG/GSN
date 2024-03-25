# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from tasks.segment_2d_patch import patch_seg

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.transformer_network import Embedder
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code, lpips, ssim
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
import imageio

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def eval(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    if args.run_val == False:
        # create training dataset
        dataset, sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=True if sampler is None else False,
        )
        iterator = iter(loader)
    else:
        # create validation dataset
        dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
        loader = DataLoader(dataset, batch_size=1)
        iterator = iter(loader)

    # Create IBRNet model
    if args.net == "trans_ibr":
        pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
    else:
        pos_enc = None
        view_enc = None
    model = IBRNetModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device, dinofield=args.dinofield)

    indx = 0
    psnr_mets = []
    lpips_mets = []
    ssim_mets = []
    while True:
        try:
            data = next(iterator)
        except:
            break
        # Rest is logging
        if args.local_rank == 0:
            tmp_ray_sampler = RaySamplerSingleImage(args, data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            if args.dinofield:
                gt_dino = tmp_ray_sampler.dino.reshape(H, W, args.dino_dim)
            else:
                gt_dino = None

            psnr_curr_img, lpips_curr_img, ssim_curr_img = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                gt_dino,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                net=args.net,
                pos_enc=pos_enc,
                view_enc=view_enc,
                transret_alpha=args.N_importance > 0,
                transsep_fine=args.transsep_fine,
                dinofield=args.dinofield,
            )
            psnr_mets.append(psnr_curr_img)
            lpips_mets.append(lpips_curr_img)
            ssim_mets.append(ssim_curr_img)
            torch.cuda.empty_cache()
            indx += 1
    print("Average PSNR: ", np.mean(psnr_mets))
    print("Average LPIPS: ", np.mean(lpips_mets))
    print("Average SSIM: ", np.mean(ssim_mets))


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    gt_dino,
    render_stride=1,
    prefix="",
    out_folder="",
    net="mlp_ibr",
    pos_enc=None,
    view_enc=None,
    transret_alpha=False,
    transsep_fine=False,
    dinofield=False,
):  
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            net=net,
            pos_enc=pos_enc,
            view_enc=view_enc,
            transret_alpha=transret_alpha,
            transsep_fine=transsep_fine,
            dinofield=dinofield,
        )

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    if dinofield:
        average_dino = ray_sampler.src_dinos.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

        if dinofield:
            average_dino = average_dino[::render_stride, ::render_stride]
            gt_dino = gt_dino[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
    
    if dinofield:
        dino_pred = img_HWC2CHW(ret["outputs_coarse"]["dino"].detach().cpu())
        dino_gt = img_HWC2CHW(gt_dino)
        average_dino = img_HWC2CHW(average_dino.squeeze(-1))

    patch_seg(ret["outputs_coarse"]["dino"].detach().cpu(), ret["outputs_coarse"]["rgb"].detach().cpu(), gt_img, "/home/sushanth/ZSGNT_AAAI/feature_extractor/pca/llff_clip_pcacake.pkl", 0.12, 'patch_seg.png', global_step, gt_dino)
    
    if dinofield:
        average_dino[:3] = (average_dino[:3] - average_dino[:3].min())/(average_dino[:3].max() - average_dino[:3].min())
        dino_gt[:3] = (dino_gt[:3] - dino_gt[:3].min())/(dino_gt[:3].max() - dino_gt[:3].min())
        dino_pred[:3] = (dino_pred[:3] - dino_pred[:3].min())/(dino_pred[:3].max() - dino_pred[:3].min())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, 1 * h_max, 2 * w_max)
    rgb_im[:, : rgb_gt.shape[-2],  : rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2],  : rgb_pred.shape[-1]] = rgb_pred
    if dinofield:
        rgb_im = torch.zeros(3, 2 * h_max, 2 * w_max)
        rgb_im[:, : rgb_gt.shape[-2],  : rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, : rgb_pred.shape[-2], w_max : w_max + rgb_pred.shape[-1]] = rgb_pred
        rgb_im[:, h_max : h_max + dino_gt.shape[-2],  :  dino_gt.shape[-1]] = dino_gt[:3]
        rgb_im[:, h_max : h_max + dino_pred.shape[-2], w_max : w_max + dino_pred.shape[-1]] = dino_pred[:3]

    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)
    
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )

    pred_rgb = torch.clip(pred_rgb, 0.0, 1.0)
    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    model.switch_to_train()
    return psnr_curr_img, lpips_curr_img, ssim_curr_img

if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)
