from dataloaders import train_dataloader, val_dataloader
from model import Dhvani


if __name__ == "__main__":
    dhvani = Dhvani()
    dataloaders = (train_dataloader, val_dataloader, None)
    dhvani.train(dataloaders)
