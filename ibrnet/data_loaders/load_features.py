import torch
import numpy as np
from tqdm import tqdm

###### utils
def interpolate_imlabel(imlabel, imh, imw):
    # inp: [C (optional), H, W]
    if type(imlabel) is np.ndarray:
        imlabel = torch.from_numpy(imlabel)
    if len(imlabel.shape) > 2:
        # for labeled image
        imlabel = imlabel[None].float()
    else:
        imlabel = imlabel[None, None].float()
    return torch.nn.functional.interpolate(imlabel, size=(imh, imw), mode="bilinear")[0]

def load_features(file, normalised=True, imhw=None, selected=None):

    try:
        features = torch.load(file)
    except:
        print(file)
        quit()
    
    # print
    
    # if features[list(features.keys())[0]].shape[0] != 64:
    #     features[list(features.keys())[0]] = features[list(features.keys())[0]].permute(2, 0, 1)
    
    ret_features = torch.zeros([*imhw, features[list(features.keys())[0]].shape[0], len(features)], dtype=torch.float32, device="cpu")

    if normalised:
        # print("Normalizing features.")
        for k in features:
            features[k] = torch.nn.functional.normalize(features[k], dim=0)
        # print("Normalized features.")

    if imhw is not None:
        # print("Upscaling features")
        for i, k in enumerate(features):
            if selected is None:
                # features[k] = interpolate_imlabel(features[k], imhw[0], imhw[1]).cpu()
                f = interpolate_imlabel(features[k], imhw[0], imhw[1]).cpu()
                ret_features[:,:,:,i] = f.permute(1, 2, 0).contiguous()
            else:
                if k in selected:
                    features[k] = interpolate_imlabel(features[k], imhw[0], imhw[1]).cpu()
        # print("Upscaled features.")

    return ret_features