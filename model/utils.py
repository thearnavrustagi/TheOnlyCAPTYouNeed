import torch
from numpy import mean
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
        return {"auc": [], "precision": [], "recall": [], "f1_score": []}

    @staticmethod
    def evaluate(metrics, *, y_pred, y, n_classes):
        for key in metrics.keys():
            val = None
            match key:
                case "auc":
                    pass
                    #val = MetricEvaluater.compute_auc(y_pred, y, n_classes)
                case "precision":
                    pass
                    #val = MetricEvaluater.compute_precision(y_pred, y, n_classes)
                case "recall":
                    pass
                    #val = MetricEvaluater.compute_recall(y_pred, y, n_classes)
                case "f1_score":
                    val = MetricEvaluater.compute_f1_score(y_pred, y, n_classes)

            metrics[key] = val

    @staticmethod
    def compute_auc(y_pred, y, n_classes):
        return -1

    # takes logits as input
    @staticmethod
    def compute_precision(y_pred, y, n_classes):
        y_pred = Softmax()(y_pred)
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        return multiclass_precision(
            labels_prediction, labels_real, num_classes=n_classes
        ).item()

    # takes logits as input
    @staticmethod
    def compute_recall(y_pred, y, n_classes):
        y_pred = Softmax(y_pred)
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        return multiclass_recall(
            labels_prediction, labels_real, num_classes=n_classes
        ).item()

    # takes logits as input
    @staticmethod
    def compute_f1_score(y_pred, y, n_classes):
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        f1_scores = 0
        for i in range(len(y_pred)):
           f1_scores +=  multiclass_f1_score(
                labels_prediction[i], labels_real[i], num_classes=n_classes
            ).item()
        return f1_scores / len(y_pred)


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
    model, dataloader, optimizer, loss_fn, *, xidx=0, yidx=1, classes=None, dirname="./logs", epoch_number=-1
):
    progress_bar = tqdm(dataloader)
    running_loss = []

    if classes:
        metrics = MetricEvaluater.create_metric_store()

    device = "mps"
    model.to(device)
    for data in progress_bar:
        X, y_old = data[xidx], data[yidx]

        optimizer.zero_grad()
        y_pred = model(X.to(device)).to("cpu")
        y = one_hot_encode(y_old)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        status_text = f"running_loss: {mean(running_loss):.4f}; "

        if classes:
            MetricEvaluater.evaluate(metrics, y_pred=y_pred, y=y, n_classes=classes)
            status_text += f"f1_score: {mean(metrics['f1_score']):.4f}"

        progress_bar.set_description(status_text)

    metrics["loss"] = mean(running_loss)

    if classes:
        pass

    with open(f"{dirname}/train_{epoch_number}.csv", "w+") as file:
        writer = csv.writer(file)
        writer.writerow(list(metrics.keys()))
        writer.writerow(list(metrics.values()))

    return tuple(metrics)


def validate_model(
    model,
    dataloader,
    loss_fn,
    *,
    xidx=0,
    yidx=1,
    epoch_number=-1,
    classes=None,
    plot=False,
    dirname="./logs/"
):
    progress_bar = tqdm(dataloader)
    running_loss = []
    device = "mps"

    if classes:
        metrics = MetricEvaluater.create_metric_store()

    model.to(device)
    for data in progress_bar:
        X, y_old = data[xidx], data[yidx]

        y_pred = model(X.to(device)).to("cpu")
        y = one_hot_encode(y_old)

        loss = loss_fn(y_pred, y)

        running_loss.append(loss.item())
        status_text = f"running_loss: {mean(running_loss):.4f}; "

        if classes:
            MetricEvaluater.evaluate(metrics, y_pred=y_pred, y=y, n_classes=classes)
            status_text += f"f1_score: {mean(metrics['f1_score']):.4f}"

        progress_bar.set_description(status_text)

    metrics["loss"] = mean(running_loss)

    if classes:
        pass

    with open(f"{dirname}/validation_{epoch_number}.csv", "w+") as file:
        writer = csv.writer(file)
        writer.writerow(list(metrics.keys()))
        writer.writerow(list(metrics.values()))
