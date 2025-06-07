import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.trainer import ImgSemSegTrainer


def remap_oxford_labels(t: torch.Tensor):
    t = t - 1  # map [1,2,3] → [0,1,2]
    t[t == -1] = 255
    return t


def get_transforms(dataset):
    if dataset == "oxford":
        input_tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        target_tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.long, scale=False),
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
            v2.Lambda(remap_oxford_labels),
        ])
    else:  # city
        input_tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        target_tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.long, scale=False),
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
        ])
    return input_tf, target_tf


def train(args):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_tf, target_tf = get_transforms(args.dataset)

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(
            root="datasets",
            split="trainval",
            target_types='segmentation',
            transform=input_tf,
            target_transform=target_tf,
            download=True
        )
        val_data = OxfordPetsCustom(
            root="datasets",
            split="test",
            target_types='segmentation',
            transform=input_tf,
            target_transform=target_tf,
            download=True
        )
        num_classes = 3
    else:
        train_data = CityscapesCustom(
            root="datasets/cityscapes_assg2",
            split="train",
            mode="fine",
            target_type='semantic',
            transform=input_tf,
            target_transform=target_tf,
        )
        val_data = CityscapesCustom(
            root="datasets/cityscapes_assg2",
            split="val",
            mode="fine",
            target_type='semantic',
            transform=input_tf,
            target_transform=target_tf,
        )
        num_classes = 19

    # Model
    segformer_core = SegFormer(num_classes=num_classes)
    model = DeepSegmenter(segformer_core)
    model.to(device)

    if args.pretrained:
        print(f"Loading pretrained encoder from {args.pretrained}")
        state = torch.load(args.pretrained, map_location="cpu")
        model.net.encoder.load_state_dict(state, strict=True)

        if args.freeze_encoder:
            model.net.encoder.requires_grad_(False)
            print("Encoder frozen")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        amsgrad=True
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgSemSegTrainer(
        model,
        optimizer,
        loss_fn,
        scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=64,
        val_frequency=2,
    )

    trainer.train()

    # Save encoder after pretraining
    if args.dataset == "city":
        torch.save(model.net.encoder.state_dict(),
                   "saved_models/city_encoder_pretrained.pt")
        print("Saved encoder weights to saved_models/city_encoder_pretrained.pt")

    trainer.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--dataset', choices=['oxford', 'city'],
                        default='oxford', help='data set')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='path to encoder weights')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=40)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    train(args)
