from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Food101


class Food101Dataset(Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.dataset = Food101(
            root=root,
            split=split,
            transform=transform,
            download=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label
