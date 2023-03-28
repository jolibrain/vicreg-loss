from collections import defaultdict

import torch
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import wandb

from .dataset import VICRegDataset
from .model import VICRegModel
from .vicreg import VICRegLoss


class VICRegTrainer:
    def __init__(self):
        self.loss = VICRegLoss(
            inv_coeff=25,
            var_coeff=15,
            cov_coeff=1,
            gamma=1,
        )
        self.model = VICRegModel(
            in_channels=1,
            n_layers=3,
            hidden_size=12,
            representation_size=24,
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

        # Create train and validation datasets.
        train_dataset = datasets.FashionMNIST("./data", train=True, download=True)
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True)
        self.train_dataset = VICRegDataset(train_dataset)
        self.test_dataset = VICRegDataset(test_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, device: str) -> dict:
        self.model.eval()

        metrics = defaultdict(list)
        for batch, _ in dataloader:
            batch = batch.to(device)
            x = self.train_dataset.augments(batch)
            y = self.train_dataset.augments(batch)
            _, z1 = self.model(x)
            _, z2 = self.model(y)

            for metric_name, metric_value in self.loss(z1, z2).items():
                metrics[metric_name].append(metric_value.cpu().item())

        mean_metrics = dict()
        for metric_name, metric_values in metrics.items():
            mean_metrics[metric_name] = sum(metric_values) / len(metric_values)

        return mean_metrics

    @torch.no_grad()
    def embeddings_table(
        self, dataloader: DataLoader, n_batches: int, device: str
    ) -> dict:
        table = {
            "columns": ["image", "label"],
            "data": [],
        }

        # Add embedding columns.
        images, _ = next(iter(dataloader))
        h, _ = self.model(images.to(device))
        for i in range(h.shape[1]):
            table["columns"].append(f"embedding_{i}")

        # Compute embeddings for the first `n_batches`
        # of the dataloader.
        for batch_count, (images, labels) in enumerate(dataloader):
            if batch_count >= n_batches:
                break

            images = images.to(device)
            h, _ = self.model(images)

            data = [
                [wandb.Image(image.cpu()), label, *embedding.cpu()]
                for image, label, embedding in zip(images, labels, h)
            ]
            table["data"].extend(data)

        return table

    def launch_training(self, n_epochs: int, device: str):
        wandb.init(project="vicreg")

        self.model.to(device)
        summary(self.model, self.train_dataset[0][0].shape, batch_dim=0, device=device)

        wandb.watch(self.model, log="all")

        for _ in tqdm(range(n_epochs), desc="Epochs"):
            self.model.train()
            for batch, _ in tqdm(self.train_loader, desc="Training"):
                batch = batch.to(device)
                x = self.train_dataset.augments(batch)
                y = self.train_dataset.augments(batch)

                _, z1 = self.model(x)
                _, z2 = self.model(y)

                metrics = self.loss(z1, z2)
                self.optimizer.zero_grad()
                metrics["loss"].backward()
                self.optimizer.step()

            metrics = self.evaluate(self.test_loader, device)
            wandb.log(metrics)

            table = self.embeddings_table(self.test_loader, 5, device)
            wandb.log(
                {
                    "embeddings": wandb.Table(
                        data=table["data"], columns=table["columns"]
                    )
                }
            )

        wandb.finish()
