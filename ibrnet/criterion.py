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

import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        if self.args.dinofield:
            pred_rgb = outputs["rgb"]
            if "mask" in outputs:
                pred_mask = outputs["mask"].float()
            else:
                pred_mask = None
            gt_rgb = ray_batch["rgb"]
            
            loss = 0.1 * img2mse(pred_rgb, gt_rgb, pred_mask)
        else:
            pred_rgb = outputs["rgb"]
            if "mask" in outputs:
                pred_mask = outputs["mask"].float()
            else:
                pred_mask = None
            gt_rgb = ray_batch["rgb"]
            
            loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        
        if self.args.dinofield:
            pred_dino = outputs["dino"]
            if "mask" in outputs:
                pred_mask = outputs["mask"].float()
            else:
                pred_mask = None
            gt_dino = ray_batch["dino"]

            loss = loss + img2mse(pred_dino, gt_dino, pred_mask)

        return loss, scalars_to_log
