from typing import Tuple

import kornia.augmentation as augmentation
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class VICRegDataset(Dataset):
    def __init__(self, labeled_dataset: Dataset):
        super().__init__()
        image_size = labeled_dataset[0][0].size

        self.dataset = labeled_dataset
        self.augments = nn.Sequential(
            augmentation.RandomGaussianBlur((3, 3), sigma=(0.1, 2)),
            augmentation.RandomGaussianNoise(mean=0, std=0.01),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomVerticalFlip(),
            augmentation.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            # augmentation.RandomPerspective(0.5, p=1.0),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        image = to_tensor(image)
        return image, label
