import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Subset
import random

CLASS_NAMES = 27  # 26 classes + 1 misc class

def get_transform():
    return T.Compose([
        T.Resize((720, 1280)),
        T.ToTensor(),
    ])

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith('.png'):
                self.image_paths.append(os.path.join(image_dir, fname))
                label_name = fname.replace('leftImg8bit', 'gtFine_labelIds')
                self.label_paths.append(os.path.join(label_dir, label_name))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label.squeeze().long()

def load_datasets(source_root):
    datasets = []
    for domain in ["BDD", "Cityscapes", "GTA", "Mapillary"]:
        image_dir = os.path.join(source_root, domain, "images")
        label_dir = os.path.join(source_root, domain, "labels")
        datasets.append(SegmentationDataset(image_dir, label_dir, get_transform()))
    return ConcatDataset(datasets)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = load_datasets(args.source_dir)
    print(f"Total dataset size: {len(full_dataset)}")

    # Random subset of 1000
    subset_size = min(50, len(full_dataset))
    random_indices = random.sample(range(len(full_dataset)), subset_size)
    dataset = torch.utils.data.Subset(full_dataset, random_indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = deeplabv3_resnet101(pretrained=False, num_classes=CLASS_NAMES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "deeplabv3_model.pth"))
    print("Training completed and model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to source_datasets_dir (with BDD, GTA, etc.)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    train(args)