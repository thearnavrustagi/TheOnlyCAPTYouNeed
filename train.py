from dataloaders import train_dataloader
from model import Dhvani


if __name__ == "__main__":
    dhvani = Dhvani()
    dataloaders = (train_dataloader, None, None)
    dhvani.train(dataloaders)