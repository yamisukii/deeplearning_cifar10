from abc import ABCMeta, abstractmethod

import torch


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.num_classes = classes if isinstance(
            classes, int) else len(classes)

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        # create an empty confusion-matrix (C × C) on CPU
        self._confmat = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if prediction.ndim != 4:
            raise ValueError("prediction must be (b,c,h,w)")
        if target.ndim != 3:
            raise ValueError("target must be (b,h,w)")
        b, c, h, w = prediction.shape
        if c != self.num_classes:
            raise ValueError(f"expected {self.num_classes} classes, got {c}")
        if target.shape != (b, h, w):
            raise ValueError("spatial dims of prediction/target mismatch")

        pred_lbl = prediction.argmax(dim=1)  # (b,h,w)

        mask = target != 255
        pred_lbl = pred_lbl[mask].flatten()
        tgt_lbl = target[mask].flatten()

        if pred_lbl.numel() == 0:
            return

            # encode pairs
        k = tgt_lbl * self.num_classes + pred_lbl
        cm = torch.bincount(k, minlength=self.num_classes *
                            self.num_classes).reshape(self.num_classes, self.num_classes)

        self._confmat += cm.to(self._confmat.device)

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIou: {self.mIoU():.2f}"

    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''

        if self._confmat.sum() == 0:
            return 0.0

        tp = torch.diag(self._confmat.to(torch.float32))
        fp = self._confmat.sum(0).float() - tp
        fn = self._confmat.sum(1).float() - tp
        denom = tp + fp + fn

        iou = torch.where(
            denom > 0, tp / denom, torch.zeros_like(tp)
        )
        return iou.mean().item()
