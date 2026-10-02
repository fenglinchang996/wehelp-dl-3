import csv
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms import v2


def check_dataset(dataloader: DataLoader):
    features, labels = next(iter(dataloader))
    img = features[0].squeeze()
    label = labels[0]
    print(features.size())
    print(labels)
    print(img)
    print(label.item())
    plt.imshow(img, cmap="gray")
    plt.show()


def get_device() -> torch.device:
    device = (
        acc
        if (acc := torch.accelerator.current_accelerator(check_available=True))
        is not None
        else torch.device("cpu")
    )
    print(f"Using {device.type} device")
    return device


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=[64, 64], antialias=True),
        v2.Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class ImageDataset(Dataset):
    def __init__(
        self, label_file: Path, image_dir: Path, transform: v2.Compose | None = None
    ) -> None:
        self.image_labels: list[tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            image_labels: list[tuple[str, str]] = []
            image_labels = [(row[0], row[1]) for row in reader]
            ordered_labels: list[str] = list(
                dict.fromkeys([item[1] for item in image_labels])
            )
            self.label_to_id = {label: idx for idx, label in enumerate(ordered_labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
            self.image_labels = [
                (image, self.label_to_id[label]) for image, label in image_labels
            ]
        self.image_dir: Path = image_dir
        self.transform: v2.Compose | None = transform

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, index: int):
        image_path = self.image_dir / self.image_labels[index][0]
        image = read_image(str(image_path))
        label = self.image_labels[index][1]
        if self.transform:
            image = self.transform(image)
        return image, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 32, 3, stride=1, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("conv2", nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("conv3", nn.Conv2d(64, 128, 3, stride=1, padding=1)),
                    ("relu3", nn.ReLU()),
                    ("pool3", nn.MaxPool2d(2, 2)),
                ]
            )
        )
        self.ann = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128 * 8 * 8, 256)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(256, 62)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ann(torch.flatten(self.cnn(x), 1))


def main():
    device = get_device()
    train_dataset = ImageDataset(
        label_file=Path("./data/handwriting/image_labels.csv"),
        image_dir=Path("./data/handwriting/augmented_train/"),
        transform=transform,
    )
    test_dataset = ImageFolder(
        root=Path("./data/handwriting/test/"), transform=transform
    )
    assert (
        train_dataset.label_to_id == test_dataset.class_to_idx
    ), "Label mappings are not the same"

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(15):
        net.train()
        running_loss = 0.0
        for idx, (batch_X, batch_e) in enumerate(train_dataloader):
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_e)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"\repoch {epoch + 1} batch {idx + 1} done", flush=True, end="")
        avg_loss = running_loss / len(train_dataloader)
        print(f" | epoch {epoch + 1} loss: {avg_loss}")
    end_time = time.time()
    print(f"duration: {end_time - start_time}")

    net.eval()
    all_predicted = []
    with torch.inference_mode():
        correct = 0
        total = 0
        for batch_X, batch_e in test_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            outputs = net(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_X.size(0)
            correct += (predicted == batch_e).sum().item()
            all_predicted.extend(predicted.cpu().tolist())
        print(f"correctness: {correct / total * 100}%")

    torch.save(net, "./cnn_net.pt")

    # log eval results
    with open("cnn_test.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "Expected", "Predicted", "Correct"])
        for (img_path, exp_idx), pred_idx in zip(test_dataset.samples, all_predicted):
            exp_label = test_dataset.classes[exp_idx]
            pred_label = test_dataset.classes[pred_idx]
            writer.writerow([img_path, exp_label, pred_label, exp_idx == pred_idx])


if __name__ == "__main__":
    main()
