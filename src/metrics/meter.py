"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-26
"""
from typing import Tuple

import torch
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import MulticlassAUROC
from torcheval.metrics import BinaryF1Score
from torcheval.metrics import MulticlassF1Score

class Meter(object):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        self.probs = []
        self.preds = []
        self.labels = []

        self.num_classes = num_classes
        if num_classes == 2:
            self.acc_metric = BinaryAccuracy()
            self.auc_metric = BinaryAUROC()
            self.f1_metric = BinaryF1Score(average="macro")
        else:
            self.acc_metric = MulticlassAccuracy(num_classes=num_classes)
            self.auc_metric = MulticlassAUROC(num_classes=num_classes)
            self.f1_metric = MulticlassF1Score(num_classes=num_classes,\
                                                            average="macro")

    def add(
        self,
        prob,
        pred,
        label
    ) -> None:
        self.preds.append(pred)
        self.labels.append(label)
        self.probs.append(prob)

    def reset(
        self
    ) -> None:
        self.probs = []
        self.preds = []
        self.labels = []

    def update(
        self
    ) -> Tuple:
        accuracy = self.acc_metric.update(self.preds, self.labels)
        self.acc_metric.reset()

        auc = self.auc_metric.update(self.probs, self.labels)
        self.auc_metric.reset()

        f1 = self.f1_metric.update(self.preds, self.labels)
        self.f1_metric.reset()
        return accuracy, auc, f1

if __name__ == "__main__":
    y_pred = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    y_true = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
    probs = torch.tensor([0.1, 0.9, 0.8, 0.3, 0.5, 0.2, 0.6, 0.7])
    meter = Meter(num_classes=2)
    for idx in range(len(y_pred)):
        meter.add(probs[idx], y_pred[idx], y_true[idx])

    accuracy, auc, f1 = meter.update()
    print(f"Accuracy: {accuracy} | AUC: {auc} | F1: {f1}")

    meter.reset()

    y_pred = torch.tensor([0, 1, 1, 2, 1, 2, 0, 1])
    y_true = torch.tensor([0, 1, 1, 2, 1, 2, 0, 1])
    probs = torch.tensor([
        [0.1, 0.9, 0.8],
        [0.3, 0.5, 0.2],
        [0.6, 0.7, 0.4],
        [0.1, 0.9, 0.8],
        [0.3, 0.5, 0.2],
        [0.6, 0.7, 0.4],
        [0.1, 0.9, 0.8],
        [0.3, 0.5, 0.2]
    ])
    meter = Meter(num_classes=3)
    for idx in range(len(y_pred)):
        meter.add(probs[idx], y_pred[idx], y_true[idx])

    accuracy, auc, f1 = meter.update()
