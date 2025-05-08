from abc import ABCMeta, abstractmethod

import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """
        # Type and shape checks
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("prediction and target must be torch.Tensors")

        if prediction.ndim != 2:
            raise ValueError(
                "prediction must be a 2D tensor of shape (batch_size, num_classes)")

        if target.ndim != 1:
            raise ValueError(
                "target must be a 1D tensor of shape (batch_size,)")

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "prediction and target must have the same batch size")

        num_classes = prediction.shape[1]
        if not torch.all((0 <= target) & (target < num_classes)):
            raise ValueError(
                "target contains values outside the valid class range")

        # Predicted class indices
        pred_classes = torch.argmax(prediction, dim=1)

        # Update counters
        self.n_total += target.size(0)
        self.n_matching += (pred_classes == target).sum().item()

        for pred, true in zip(pred_classes, target):
            true_label = self.classes[true.item()]
            if pred == true:
                self.correct_pred[true_label] += 1
            self.total_pred[true_label] += 1

    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """
        acc = self.accuracy() * 100
        per_class_acc = self.per_class_accuracy() * 100
        return f"Accuracy: {acc:.2f}% | Per-Class Accuracy: {per_class_acc:.2f}%"

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        if self.n_total == 0:
            return 0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        accs = []
        for classname in self.classes:
            total = self.total_pred[classname]
            correct = self.correct_pred[classname]
            if total == 0:
                accs.append(0)
            else:
                accs.append(correct / total)
        return sum(accs) / len(accs)
