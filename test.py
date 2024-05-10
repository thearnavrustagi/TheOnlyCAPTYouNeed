from dataloaders import create_k_fold_dataloaders
from model import Dhvani
import torch

if __name__ == "__main__":
    k_folds = 10
    random_seed = 42
    checkpoint_path = 'saved_models/PRN_epoch-99_fold-1.pth'
    dataloaders = create_k_fold_dataloaders(k_folds=k_folds, random_seed=random_seed)
    for fold, (train_dataloader, val_dataloader, test_dataloader) in enumerate(dataloaders[2:], start=3):
        dhvani = Dhvani()

        dhvani.test_model(
            test_dataloader, fold_no=fold
        )
        break