import torch
from numpy import mean
from tqdm import tqdm
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_recall,
    multiclass_precision,
)
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
            match key:
                case "auc":
                    val = MetricEvaluater.compute_auc(y_pred, y, n_classes)
                case "precision":
                    val = MetricEvaluater.compute_precision(y_pred, y, n_classes)
                case "recall":
                    val = MetricEvaluater.compute_recall(y_pred, y, n_classes)
                case "f1_score":
                    val = MetricEvaluater.compute_f1_score(y_pred, y, n_classes)

            metrics[key] = val

    @staticmethod
    def compute_auc(y_pred, y, n_classes):
        return -1

    # takes logits as input
    @staticmethod
    def compute_precision(y_pred, y, n_classes):
        y_pred = Softmax(y_pred)
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
        y_pred = Softmax(y_pred)
        labels_prediction = torch.argmax(y_pred, axis=-1)
        labels_real = torch.argmax(y, axis=-1)
        return multiclass_f1_score(
            labels_prediction, labels_real, num_classes=n_classes
        ).item()


"""
xidx: the index of the x (inputs) to the models
yidx: the index of the expected outputs to the models
"""


def train_one_epoch(
    model, dataloader, optimizer, loss_fn, *, xidx=0, yidx=1, classes=None
):
    progress_bar = tqdm(enumerate(dataloader))
    running_loss = []

    if classes:
        metrics = MetricEvaluater.create_metric_store()

    for i, data in progress_bar:
        X, y = data[xidx], data[yidx]

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        status_text = f"running_loss: {mean(running_loss)}; "

        if classes:
            MetricEvaluater.evaluate(metrics)

        progress_bar.set_description(status_text)

    metrics.append(mean(running_loss))

    if classes:
        pass

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
):
    progress_bar = tqdm(enumerate(dataloader))
    running_loss = []
    metrics = [{"f1_score": []}]

    for i, data in progress_bar:
        X, y = data[xidx], data[yidx]
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        running_loss.append(loss.item())

        status_text = f"running_loss: {mean(running_loss)}; "

        if classes != None:
            f1_score = multiclass_f1_score(y_pred, y, num_classes=classes)
            metrics["f1_score"].append(f1_score)
            status_text += f"f1-score: {f1_score}; "

        progress_bar.set_description(status_text)

    metrics["loss"] = running_loss

    dirname = "./logs/epoch_{epoch_number}"
    os.mkdir(dirname)

    with open(f"{dirname}/validation_{epoch_number}.csv", "w+") as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerows(metrics)
