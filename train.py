from dataloaders import create_k_fold_dataloaders
from model import Dhvani

if __name__ == "__main__":
    dhvani = Dhvani()
    print(sum(p.numel() for p in dhvani.parameters()))
    k_folds = 10
    random_seed = 42
    dataloaders = create_k_fold_dataloaders(k_folds=k_folds, random_seed=random_seed)
    for fold, (train_dataloader, val_dataloader) in enumerate(dataloaders, start=1):
        print(f"Training on Fold {fold}/{k_folds}")
        dhvani.train(
            (train_dataloader, val_dataloader, None), fold_no=fold, training_mdn=False
        )
