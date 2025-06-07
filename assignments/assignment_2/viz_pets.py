import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter


def save_grid(tensor, fname, nrow=4, is_mask=False):
    if is_mask:
        tensor = tensor.float().unsqueeze(1) / tensor.max()
        grid = torchvision.utils.make_grid(tensor, nrow=nrow)
        grid = grid.expand(3, -1, -1)
    else:
        grid = torchvision.utils.make_grid(tensor, nrow=nrow)

    plt.imsave(fname, np.transpose(grid.cpu().numpy(), (1, 2, 0)))


def load_model(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location='cpu')

    if "net" in obj and isinstance(obj["net"], (dict, OrderedDict)):
        obj = obj["net"]

    model = DeepSegmenter(SegFormer(num_classes=3)).to(device)
    model.load_state_dict(obj, strict=False)
    model.eval()
    return model


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    img_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
    ])

    # Load data
    val_ds = OxfordPetsCustom(root=args.root, split="test",
                              target_types="segmentation",
                              transform=img_tf, target_transform=tgt_tf,
                              download=False)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.n,
                                         shuffle=True, num_workers=2)

    model = load_model(args.ckpt, device)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    preds = model(imgs).argmax(1)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    imgs_vis = torch.clamp(imgs * std + mean, 0, 1)

    Path("img").mkdir(exist_ok=True)
    save_grid(imgs_vis.cpu(), "img/val_inputs.png", nrow=4)
    save_grid(preds.cpu(), "img/val_preds.png", nrow=4, is_mask=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--root", default="datasets",
                    help="OxfordPets dataset root folder")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of samples to visualize")
    args = ap.parse_args()
    main(args)
