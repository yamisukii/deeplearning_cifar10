import os
from typing import Tuple

import numpy as np
from assignment_1_code.datasets.dataset import ClassificationDataset, Subset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
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

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_batch(self, filepath: str):
        with open(filepath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            data = batch[b"data"]  # shape: (10000, 3072)
            labels = batch[b"labels"]  # list of 10000 ints

            # reshape to (10000, 3, 32, 32) -> transpose to (10000, 32, 32, 3)
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return data, labels

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """
        if not os.path.isdir(self.fdir):
            raise ValueError(f"{self.fdir} is not a directory.")

        if self.subset == Subset.TRAINING:
            files = [f"data_batch_{i}" for i in range(1, 5)]
        elif self.subset == Subset.VALIDATION:
            files = ["data_batch_5"]
        elif self.subset == Subset.TEST:
            files = ["test_batch"]
        else:
            raise ValueError("Invalid subset")

        all_images, all_labels = [], []
        for fname in files:
            path = os.path.join(self.fdir, fname)
            if not os.path.exists(path):
                raise ValueError(f"Missing CIFAR-10 file: {path}")

            data, labels = self.load_batch(path)
            all_images.append(data)
            all_labels.extend(labels)

        return np.concatenate(all_images), np.array(all_labels)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self.labels):
            raise IndexError("Index out of bounds")

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)
