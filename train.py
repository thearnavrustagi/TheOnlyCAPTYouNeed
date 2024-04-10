from dataloaders import L1SpeechDataset

if __name__ == "__main__":
    l1 = L1SpeechDataset()
    print(l1[3])
