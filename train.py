from dataloaders import train_dataloader
from model import MelSpectrogramEncoder

if __name__ == "__main__":
    ms, e_p, tokens =  next(iter(train_dataloader))
    print(ms.shape)
    layer = MelSpectrogramEncoder()
    y = layer(ms)
    print(y.shape)
