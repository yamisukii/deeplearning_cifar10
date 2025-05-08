import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
from assignment_1_code.datasets.dataset import Subset

from dlvc_ss25.assignments.assignment_1.assignment_1_code.datasets.cifar10 import \
    CIFAR10Dataset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_data = CIFAR10Dataset(
        fdir="dlvc_ss25/assignments/assignment_1/cifar-10-batches-py",
        subset=Subset.TRAINING,
        transform=transform,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=2
    )

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))
