import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchinfo import summary

from .dataset import VICRegDataset
from .model import VICRegModel
from .vicreg import VICRegLoss


class VICRegTrainer:
    def __init__(self):
        self.loss = VICRegLoss()
        self.model = VICRegModel(1, 3, 12, 24)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

        # Create train and validation datasets.
        train_dataset = datasets.FashionMNIST("./data", train=True, download=True)
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True)
        self.train_dataset = VICRegDataset(train_dataset)
        self.test_dataset = VICRegDataset(test_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def launch_training(self, n_epochs: int, device: str):
        self.model.to(device)
        self.model.train()

        summary(self.model, self.train_dataset[0].shape, batch_dim=0, device=device)

        for _ in range(n_epochs):
            for batch in self.train_loader:
                batch = batch.to(device)
                x = self.train_dataset.augments(batch)
                y = self.train_dataset.augments(batch)

                x = self.model(x)
                y = self.model(y)

                metrics = self.loss(x, y)
                self.optimizer.zero_grad()
                metrics["loss"].backward()
                self.optimizer.step()

                print(metrics["loss"].item())
