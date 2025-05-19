# TODO:
#  build_dataset(...) -> SequenceDataset
#  build_dataloaders(batch_size=64, split=(0.8, 0.1, 0.1)) -> train_loader, val_loader, test_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class SequenceDataset(Dataset):
    def __init__(self, data, labels, masks):
        self.data = data
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx], self.labels[idx]

def build_dataset(data, labels, masks):
    return SequenceDataset(data, labels, masks)

def build_dataloaders(data, labels, masks, batch_size=64, split=(0.8, 0.1, 0.1)):
    dataset = build_dataset(data, labels, masks)

    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from data.generators import generate_data

    data, labels, masks = generate_data(num_samples=1000)

    train_loader, val_loader, test_loader = build_dataloaders(data, labels, masks)

    print(next(iter(train_loader)))
