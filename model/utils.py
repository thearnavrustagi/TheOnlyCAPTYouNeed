import torch
from numpy import mean
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_f1_score
import csv
import os

"""
xidx: the index of the x (inputs) to the models
yidx: the index of the expected outputs to the models
"""
def train_one_epoch(model, dataloader, optimizer, loss_fn, *, xidx=0, yidx=1, classes=None):
    progress_bar = tqdm(enumerate(dataloader))
    running_loss = []
    metrics = []

    if classes:
        metrics["auc"] = []
        metrics["precision"] = []
        metrics["recall"] = []
        metrics["f1_score"] = []


    for  i, data in progress_bar:
        X, y = data[xidx], data[yidx]

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        status_text = f"running_loss: {mean(running_loss)}; "
        
        if classes:
            f1_score = multiclass_f1_score(y_pred, y, num_classes=classes)
            metrics["f1_score"].append()
            status_text += f"f1-score: {f1_score}; "

        progress_bar.set_description(status_text)

    metrics.append(mean(running_loss))

    if classes:

    
    return tuple(metrics)

def validate_model(model, dataloader, loss_fn, *, xidx=0, yidx=1, epoch_number=-1, classes=None, plot=False):
    progress_bar = tqdm(enumerate(dataloader))
    running_loss = []
    metrics = [{"f1_score":[]}]

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