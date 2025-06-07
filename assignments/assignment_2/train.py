
import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.models.segment_model import DeepSegmenter
from dlvc.trainer import ImgSemSegTrainer
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


def pets_label_shift(mask: torch.Tensor) -> torch.Tensor:
    """1,2,3→0,1,2 """
    mask = mask.squeeze(0)
    void = mask == 255
    mask = mask - 1
    mask[void] = 255
    return mask


def train(args):
    mask_transform_pets = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Lambda(pets_label_shift),
    ])

    mask_transform_city = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Lambda(lambda t: t.squeeze(0)),
    ])

    train_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(
                                      size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(),
                                   v2.ToDtype(torch.long, scale=False),
                                   v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])  # ,

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Resize(
                                    size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.long, scale=False),
                                 v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="datasets",
                                  split="trainval",
                                  target_types='segmentation',
                                  transform=train_transform,
                                  target_transform=mask_transform_pets,
                                  download=True)

    val_data = OxfordPetsCustom(root="datasets",
                                split="test",
                                target_types='segmentation',
                                transform=val_transform,
                                target_transform=mask_transform_pets,
                                download=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch_model = fcn_resnet50(
        num_classes=3,
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT
    )
    model = DeepSegmenter(torch_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.98
    )

    train_metric = SegMetrics(classes=3)
    val_metric = SegMetrics(classes=3)
    # train_metric = SegMetrics(classes=train_data.classes_seg)
    # val_metric = SegMetrics(classes=val_data.classes_seg)

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgSemSegTrainer(model,
                               optimizer,
                               loss_fn,
                               lr_scheduler,
                               train_metric,
                               val_metric,
                               train_data,
                               val_data,
                               device,
                               args.num_epochs,
                               model_save_dir,
                               batch_size=64,
                               val_frequency=2)
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)
