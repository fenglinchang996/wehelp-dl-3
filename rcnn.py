import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes

WEIGHTS_PATH = Path("rcnn_model.pth")


def draw_image_with_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: list[str] | None = None,
    colors: str = "red",
    save_path: Path | str | None = None,
    show=False,
):
    uint8_image = (image * 255).to(torch.uint8) if image.dtype != torch.uint8 else image
    uint8_image = uint8_image.cpu()
    boxes = boxes.cpu()

    annotated_image = draw_bounding_boxes(
        image=uint8_image,
        boxes=boxes,
        labels=labels,
        colors=colors,
        width=2,
    )

    plt.figure(figsize=(30, 30))
    plt.imshow(annotated_image.permute(1, 2, 0))
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def check_dataset(dataset: VehicleDataset):
    image, target = dataset[0]
    print("Dataset size: ", len(dataset))
    print("Image: ", image.shape, "dtype:", image.dtype)
    print("Box shape: ", target["boxes"].shape, "Labels: ", target["labels"])
    labels = [dataset.id_to_class[int(label)] for label in target["labels"]]
    draw_image_with_bounding_boxes(image, boxes=target["boxes"], labels=labels)


def get_device() -> torch.device:
    device = (
        acc
        if (acc := torch.accelerator.current_accelerator(check_available=True))
        is not None
        else torch.device("cpu")
    )
    print(f"Using {device.type} device")
    return device


def get_category_mapping():
    category = {"Bus": 0, "Car": 1, "Motorcycle": 2, "Pickup": 3, "Truck": 4}
    category_to_id = {
        key: value + 1 for key, value in category.items()
    }  # 0 for background
    id_to_category = {value: key for key, value in category_to_id.items()}
    return category_to_id, id_to_category


transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


class VehicleDataset(Dataset):
    def __init__(
        self, label_file: Path, image_dir: Path, transform: T.Compose | None
    ) -> None:
        super().__init__()
        category_to_id, id_to_category = get_category_mapping()
        self.class_num = len(id_to_category) + 1
        self.class_to_id = category_to_id
        self.id_to_class = id_to_category
        self.image_dir: Path = image_dir
        self.transform: T.Compose | None = transform
        self.image_label_bboxes = []
        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            image_label_bboxes_dict: dict[
                str, list[tuple[int, tuple[int, int, int, int]]]
            ] = {}
            for row in reader:
                image_path = row[0]
                class_name = row[3]
                bounding_box = (int(row[4]), int(row[5]), int(row[6]), int(row[7]))
                if image_path not in image_label_bboxes_dict:
                    image_label_bboxes_dict[image_path] = []
                image_label_bboxes_dict[image_path].append(
                    (self.class_to_id[class_name], bounding_box)
                )

            self.image_label_bboxes = [
                (key, value) for key, value in image_label_bboxes_dict.items()
            ]

    def __len__(self) -> int:
        return len(self.image_label_bboxes)

    def __getitem__(self, index: int):
        image_id = index
        image_path = self.image_dir / self.image_label_bboxes[index][0]
        image = read_image(str(image_path))
        if self.transform:
            image = self.transform(image)
        image = tv_tensors.Image(image)
        image_size = T.functional.get_size(image)
        H, W = image_size[0], image_size[1]
        target = {}
        target["image_id"] = image_id
        label_box_pairs = self.image_label_bboxes[index][1]
        target["boxes"] = tv_tensors.BoundingBoxes(
            [box for _, box in label_box_pairs],
            format="XYXY",
            canvas_size=(H, W),
            dtype=torch.float32,
        )  # type: ignore
        target["labels"] = torch.tensor(
            [label for label, _ in label_box_pairs], dtype=torch.int64
        )
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int, weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    assert isinstance(model.roi_heads.box_predictor, FastRCNNPredictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_model(model: FasterRCNN, dataset: VehicleDataset, device: torch.device):
    train_dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    nums_epochs = 5
    start_time = time.time()
    for epoch in range(nums_epochs):
        model.train()
        running_loss = 0.0
        for idx, (images, targets) in enumerate(train_dataloader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]
            loss_dict: dict[str, torch.Tensor] = model(images, targets)
            losses: torch.Tensor = sum(loss for loss in loss_dict.values())  # type: ignore
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
            print(f"\repoch {epoch + 1} batch {idx + 1} done", flush=True, end="")
        avg_loss = running_loss / len(train_dataloader)
        print(f" | epoch {epoch + 1} loss: {avg_loss}")
    end_time = time.time()
    print(f"duration: {end_time - start_time}")
    torch.save(model.state_dict(), WEIGHTS_PATH)


def main():
    device = get_device()
    model = None

    train_dataset = VehicleDataset(
        label_file=Path("data/vehicles_images/train_labels.csv"),
        image_dir=Path("data/vehicles_images/train"),
        transform=transform,
    )

    # set True for running training session
    if False:
        model = get_model(train_dataset.class_num)
        train_model(model, dataset=train_dataset, device=device)

    test_dataset = VehicleDataset(
        label_file=Path("data/vehicles_images/test_labels.csv"),
        image_dir=Path("data/vehicles_images/test"),
        transform=transform,
    )
    if not model:
        model = get_model(test_dataset.class_num).to(device)
        model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )

    test_dataloader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    with torch.inference_mode():
        detected_scores: list[float] = []
        for idx, (images, targets) in enumerate(test_dataloader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]
            outputs = model(images)
            for output in outputs:
                threshold = 0.45
                scores = output["scores"]
                testing_scores = scores[scores >= threshold]
                detected_scores.extend(testing_scores.cpu().tolist())
            if idx == 0:
                for image, target, output in zip(images, targets, outputs):
                    output_dir = Path("rcnn_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_id = target["image_id"]
                    expected_boxes = target["boxes"]
                    expected_labels = [
                        test_dataset.id_to_class[label.item()]  # type: ignore
                        for label in target["labels"]
                    ]
                    draw_image_with_bounding_boxes(
                        image=image,
                        boxes=expected_boxes,
                        labels=expected_labels,
                        save_path=output_dir / f"expected_{image_id}.png",
                        show=False,
                    )
                    threshold = 0.45
                    scores = output["scores"]
                    mask = scores >= threshold
                    testing_boxes = output["boxes"][mask]
                    testing_labels = [
                        test_dataset.id_to_class[label.item()]
                        for label in output["labels"][mask]
                    ]
                    draw_image_with_bounding_boxes(
                        image=image,
                        boxes=testing_boxes,
                        labels=testing_labels,
                        save_path=output_dir / f"detected_{image_id}.png",
                        show=False,
                    )

        print(f"avg score: {sum(detected_scores) / len(detected_scores) * 100}%")


if __name__ == "__main__":
    main()
