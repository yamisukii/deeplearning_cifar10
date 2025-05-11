# Feel free to change the imports according to your implementation and needs
import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from torchvision.models import resnet18  # change to the model you want to test


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test data
    data_dir = Path("dlvc_ss25/assignments/assignment_1/cifar-10-batches-py")
    test_data = CIFAR10Dataset(
        fdir=data_dir, subset=Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False)

    num_test_data = len(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = resnet18(num_classes=len(test_data.classes))
    model = DeepClassifier(net)
    model.load(args.path_to_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    # Below implement testing loop and print final loss
    # and metrics to terminal after testing is finished
    # ...
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_metric.update(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_data_loader)
    acc = test_metric.accuracy()
    pc_acc = test_metric.per_class_accuracy()

    print(f"\nTest Results")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Per-class Accuracy: {pc_acc:.4f}")

    for class_name in test_metric.classes:
        total = test_metric.total_pred[class_name]
        correct = test_metric.correct_pred[class_name]
        class_acc = correct / total if total > 0 else 0.0
        print(f"Accuracy for class: {class_name:<6} is {class_acc:.2f}")


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    )
    args.add_argument(
        "--path_to_trained_model", required=True, type=str,
        help="Path to the trained model (.pth file)"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.path_to_trained_model = "dlvc_ss25/saved_models/model_best.pth"

    test(args)
