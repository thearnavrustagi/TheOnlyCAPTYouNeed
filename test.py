from dataloaders import train_dataloader, val_dataloader
from model import Dhvani


if __name__ == "__main__":
    for ms, err_p, tokens in train_dataloader:
        print(ms.shape)
        print(err_p.shape)
        print(tokens.shape)
