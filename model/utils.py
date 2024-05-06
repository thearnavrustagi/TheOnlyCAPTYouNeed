import torch
from numpy import mean
import numpy as np
from tqdm import tqdm
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_recall,
    multiclass_precision,
)
from .hyperparameters import (
    PRN_CLF_OUT_DIM,
)
from torch.nn.functional import one_hot
from torch.nn import Softmax
import csv
import os


class MetricEvaluater(object):
    @staticmethod
    def create_metric_store():
        return {"precision": [], "recall": [], "f1_score": [], "accuracy": []}

    @staticmethod
    def evaluate(metrics, *, y_pred, y, n_classes):
        for key in metrics.keys():
            val = None
            match key:
                case "auc":
                    pass
                    metrics[key].append(
                        MetricEvaluater.compute_auc(y_pred, y, n_classes)
                    )
                case "precision":
                    pass
                    metrics[key].append(
                        MetricEvaluater.compute_precision(y_pred, y, n_classes)
                    )
                case "recall":
                    pass
                    metrics[key].append(
                        MetricEvaluater.compute_recall(y_pred, y, n_classes)
                    )
                case "f1_score":
                    pass
                    metrics[key].append(
                        MetricEvaluater.compute_f1_score(y_pred, y, n_classes)
                    )
                case "accuracy":
                    metrics[key].append(
                        MetricEvaluater.compute_accuracy(y_pred, y, n_classes)
                    )

    @staticmethod
    def compute_auc(y_pred, y, n_classes):
        return -1

    # takes logits as input
    @staticmethod
    def compute_precision(y_pred, y, n_classes):
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        precisions = 0
        for i in range(len(y_pred)):
            precisions += multiclass_precision(
                labels_prediction[i], labels_real[i], num_classes=n_classes
            ).item()
        return precisions / y_pred.shape[0]

    # takes logits as input
    @staticmethod
    def compute_recall(y_pred, y, n_classes):
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        recalls = 0
        for i in range(len(y_pred)):
            recalls += multiclass_recall(
                labels_prediction[i], labels_real[i], num_classes=n_classes
            ).item()
        return recalls / y_pred.shape[0]

    # takes logits as input
    @staticmethod
    def compute_f1_score(y_pred, y, n_classes):
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        f1_scores = 0
        for i in range(len(y_pred)):
            f1_scores += multiclass_f1_score(
                labels_prediction[i], labels_real[i], num_classes=n_classes
            ).item()
        return f1_scores / y_pred.shape[0]

    def compute_accuracy(y_pred, y, n_classes):
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        precisions = 0
        mask = labels_prediction == labels_real
        return np.mean(mask.numpy())


def one_hot_encode(index):
    unique_classes = PRN_CLF_OUT_DIM
    output_size = (index.shape[0], index.shape[1], unique_classes)

    one_hot_tensor = torch.zeros(output_size)
    index = index.unsqueeze(dim=2)
    one_hot_tensor.scatter_(-1, index, 1)

    return one_hot_tensor


"""
xidx: the index of the x (inputs) to the models
yidx: the index of the expected outputs to the models
"""


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    *,
    xidx=0,
    yidx=1,
    classes=None,
    dirname="./logs",
    model_dir="./saved_models",
    epoch_number=-1,
    train=True,
    optimizer=None,
    fold=1,
):
    progress_bar = tqdm(dataloader)
    running_loss = []

    if classes:
        metrics = MetricEvaluater.create_metric_store()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    for data in progress_bar:
        X, y_old = data[xidx], data[yidx]

        y_pred = model(X.to(device)).to("cpu")
        y = one_hot_encode(y_old)

        loss = loss_fn(y_pred, y)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss.append(loss.item())
        status_text = f"running_loss: {mean(running_loss):.4f}; "

        if classes:
            MetricEvaluater.evaluate(metrics, y_pred=y_pred, y=y, n_classes=classes)
            status_text += f"f1_score: {mean(metrics['f1_score']):.4f}; "
            status_text += f"recall: {mean(metrics['recall']):.4f}; "
            status_text += f"precision: {mean(metrics['precision']):.4f} "
            status_text += f"accuracy: {mean(metrics['accuracy']):.4f} "
        progress_bar.set_description(status_text)

    metrics["loss"] = running_loss
    checkpoint = {
        "epoch": epoch_number,
        "fold": fold,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{model_dir}/model_epoch-{epoch_number}_fold-{fold}.pth")

    if classes:
        pass

    with open(
        f"{dirname}/{'train' if train else 'validation'}_epoch-{epoch_number}_fold-{fold}.csv",
        "w+",
    ) as file:
        writer = csv.writer(file)
        keys = list(metrics.keys())
        vals = [[] for _ in range(len(keys))]
        writer.writerow(keys)
        for key, val in metrics.items():
            vals[keys.index(key)] = val
        writer.writerows(np.column_stack(vals))

    return tuple(metrics)
