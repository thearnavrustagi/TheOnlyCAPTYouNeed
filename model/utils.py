import torch
from numpy import mean
from torcheval.metrics.functional import multiclass_f1_score

"""
xidx: the index of the x (inputs) to the models
yidx: 
"""


def train_one_epoch(
    model, dataloader, optimizer, loss_fn, *, xidx=-0, yidx=1, classes=None
):
    progress_bar = tqdm(enumerate(dataloader))
    running_loss = []
    for i, data in progress_bar:
        X, y = data[xidx], data[yidx]

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        if classes != None:
            f1_score = multiclass_f1_score(y_pred, y, num_classes=classes)
            status_text += " f1-score: {f1_score}"
        loss.backward()

        optimizer.step()
        running_loss.append(loss.item())

        progress_bar.set_description(f"loss: {mean(running_loss)}; f1-score: ")
