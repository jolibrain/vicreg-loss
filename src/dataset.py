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
            augmentation.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0)),
            augmentation.RandomGaussianNoise(),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomVerticalFlip(),
            augmentation.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        image = to_tensor(image)
        return image
