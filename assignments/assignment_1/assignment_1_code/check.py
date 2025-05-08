import torch
from assignment_1_code.metrics import Accuracy

classes = ["cat", "dog", "horse"]
acc = Accuracy(classes)

prediction = torch.tensor([
    [0.2, 0.3, 0.5],  # pred: horse
    [0.6, 0.1, 0.3],  # pred: cat
    [0.1, 0.8, 0.1],  # pred: dog
])
target = torch.tensor([2, 0, 2])  # true: horse, cat, horse

acc.update(prediction, target)
print(str(acc))  # Expect: Accuracy: 66.67%, Per-Class Accuracy: ~66.67%
