import os
import random
import numpy as np
import albumentations as A
import albumentations.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from PIL import Image
from collections import Counter


class OrthoXRayDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.samples = []
        class_names = os.listdir(root)
        class_names = [c for c in class_names if os.path.isdir(os.path.join(root, c))]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_names))}
        for cls_name in sorted(class_names):
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    img_path = os.path.join(cls_dir, fname)
                    label_id = self.class_to_idx[cls_name]
                    self.samples.append((img_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = np.array(image, dtype=np.uint8)
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = np.array(image, dtype=np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        return image, label


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1    = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2    = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


from torchvision.models.resnet import Bottleneck

class CBAMBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, *args, ratio=16, kernel_size=7, **kwargs):
        super().__init__(inplanes, planes, *args, **kwargs)
        self.cbam = CBAM(self.conv3.out_channels, ratio=ratio, kernel_size=kernel_size)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def _replace_layer_with_cbam(layer, ratio=16, kernel_size=7):
    new_blocks = []
    for b in layer:
        inplanes = b.conv1.in_channels
        planes = b.conv3.out_channels // b.expansion
        cbam_b = CBAMBottleneck(
            inplanes=inplanes,
            planes=planes,
            stride=b.stride,
            downsample=b.downsample,
            groups=getattr(b, 'groups', 1),
            base_width=getattr(b, 'base_width', 64),
            dilation=getattr(b, 'dilation', 1),
            ratio=ratio,
            kernel_size=kernel_size
        )
        cbam_b.conv1 = b.conv1
        cbam_b.bn1 = b.bn1
        cbam_b.conv2 = b.conv2
        cbam_b.bn2 = b.bn2
        cbam_b.conv3 = b.conv3
        cbam_b.bn3 = b.bn3
        cbam_b.relu = b.relu
        cbam_b.expansion = b.expansion
        new_blocks.append(cbam_b)
    return nn.Sequential(*new_blocks)

def build_cbam_resnet50(num_classes=6, pretrained=True, ratio=16, kernel_size=7, dropout_rate=0.5):
    resnet50 = models.resnet50(pretrained=pretrained)
    resnet50.layer1 = _replace_layer_with_cbam(resnet50.layer1, ratio=ratio, kernel_size=kernel_size)
    resnet50.layer2 = _replace_layer_with_cbam(resnet50.layer2, ratio=ratio, kernel_size=kernel_size)
    resnet50.layer3 = _replace_layer_with_cbam(resnet50.layer3, ratio=ratio, kernel_size=kernel_size)
    resnet50.layer4 = _replace_layer_with_cbam(resnet50.layer4, ratio=ratio, kernel_size=kernel_size)
    in_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return resnet50


def get_augmentations(mean, std, is_train=True, augment_level='medium'):
    if is_train:
        if augment_level == 'light':
            transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                A.pytorch.transforms.ToTensorV2()
            ])
        elif augment_level == 'medium':
            transform = A.Compose([
                A.Resize(224, 224),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
                A.Normalize(mean=mean, std=std),
                A.pytorch.transforms.ToTensorV2()
            ])
        elif augment_level == 'strong':
            transform = A.Compose([
                A.Resize(224, 224),
                A.Rotate(limit=30, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.GaussNoise(var_limit=(20.0, 80.0), p=0.7),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.7),
                A.CoarseDropout(max_holes=16, max_height=32, max_width=32, fill_value=0, p=0.7),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(60, 140), p=0.5),
                A.Normalize(mean=mean, std=std),
                A.pytorch.transforms.ToTensorV2()
            ])
        else:
            transform = get_augmentations(mean, std, is_train=True, augment_level='medium')
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std),
            A.pytorch.transforms.ToTensorV2()
        ])
    return transform


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    mean = 0.0
    std = 0.0
    total = 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += images.size(0)
    mean /= total
    std /= total
    return mean.tolist(), std.tolist()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.cls = classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.topk(outputs, 3, dim=1)  
        correct_top1 += (preds[:, 0] == labels).sum().item()  
        correct_top2 += (preds[:, :2] == labels.view(-1, 1)).any(dim=1).sum().item() 
        correct_top3 += (preds[:, :3] == labels.view(-1, 1)).any(dim=1).sum().item()  
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc_top1 = correct_top1 / total
    train_acc_top2 = correct_top2 / total
    train_acc_top3 = correct_top3 / total
    return train_loss, train_acc_top1, train_acc_top2, train_acc_top3

def val_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.topk(outputs, 3, dim=1)  
            correct_top1 += (preds[:, 0] == labels).sum().item()  # Top-1
            correct_top2 += (preds[:, :2] == labels.view(-1, 1)).any(dim=1).sum().item()  # Top-2
            correct_top3 += (preds[:, :3] == labels.view(-1, 1)).any(dim=1).sum().item()  # Top-3
            total += labels.size(0)
    
    val_loss = running_loss / total
    val_acc_top1 = correct_top1 / total
    val_acc_top2 = correct_top2 / total
    val_acc_top3 = correct_top3 / total
    return val_loss, val_acc_top1, val_acc_top2, val_acc_top3

def train_val_split_by_class(dataset, val_ratio=0.2, seed=42):
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    train_idx, val_idx = [], []
    random.seed(seed)
    for label, idx_list in class_indices.items():
        random.shuffle(idx_list)
        split = int(len(idx_list) * val_ratio)
        val_idx.extend(idx_list[:split])
        train_idx.extend(idx_list[split:])
    return train_idx, val_idx


def main(
    model_name='cbam_resnet50',
    pretrained=True,
    cbam_ratio=16,
    cbam_kernel_size=7,
    dropout_rate=0.5,
    optimizer_type='AdamW',
    lr_phase1=1e-4,
    lr_phase2=1e-5,
    weight_decay=1e-4,
    batch_size=32,
    augment_level='medium',
    label_smoothing_epsilon=0.1,
    early_stopping_patience_phase1=80,
    early_stopping_patience_phase2=200,
    phase1_epochs=5,
    phase2_epochs=50
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = "final_dataset_all"  
    train_root = os.path.join(data_root, "train")

    # 加载 train 数据集
    train_dataset_full = OrthoXRayDataset(train_root)
    num_classes = len(train_dataset_full.class_to_idx)

    print("========== Dataset Analysis ==========")
    print("Full Train dataset class distribution:", Counter([label for _, label in train_dataset_full.samples]))

    print("\nComputing mean & std from FULL training set...")
    train_mean, train_std = compute_mean_std(train_dataset_full)
    val_mean, val_std = train_mean, train_std
    print(f"Train mean={train_mean}, std={train_std}")
    print(f"Val   mean={val_mean}, std={val_std}")

    train_transform = get_augmentations(train_mean, train_std, is_train=True, augment_level=augment_level)
    val_transform = get_augmentations(val_mean, val_std, is_train=False)

    # 划分训练集和验证集
    train_indices, val_indices = train_val_split_by_class(train_dataset_full, val_ratio=0.2, seed=42)
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(train_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("\nFinal Train Dataset size:", len(train_dataset))
    print("Final Val Dataset size:", len(val_dataset))

    # 保存训练集和验证集图片路径
    with open('train_image_paths.txt', 'w') as f:
        for idx in train_indices:
            img_path, _ = train_dataset_full.samples[idx]
            f.write(img_path + '\n')

    with open('val_image_paths.txt', 'w') as f:
        for idx in val_indices:
            img_path, _ = train_dataset_full.samples[idx]
            f.write(img_path + '\n')

    print(f"\n========== Building Model: {model_name} ==========")
    if model_name == 'cbam_resnet50':
        model = build_cbam_resnet50(num_classes=num_classes, pretrained=pretrained, ratio=cbam_ratio, kernel_size=cbam_kernel_size, dropout_rate=dropout_rate)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.to(device)

    print(f"\n========== Optimizer: {optimizer_type} ==========")
    if optimizer_type == 'AdamW':
        optimizer_phase1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_phase1, weight_decay=weight_decay)
        optimizer_phase2 = torch.optim.AdamW(model.parameters(), lr=lr_phase2, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer_phase1 = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_phase1, momentum=0.9, weight_decay=weight_decay)
        optimizer_phase2 = torch.optim.SGD(model.parameters(), lr=lr_phase2, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    scheduler_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=phase1_epochs)
    scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=phase2_epochs)

    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=label_smoothing_epsilon)

    early_stopping_phase1 = EarlyStopping(patience=early_stopping_patience_phase1, verbose=True, path=f'best_{model_name}_phase1.pt')
    early_stopping_phase2 = EarlyStopping(patience=early_stopping_patience_phase2, verbose=True, path=f'best_{model_name}_phase2.pt')

    best_acc_phase1 = 0.0
    print("\n========== Phase 1: Train last layers only ==========")
    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc_top1, train_acc_top2, train_acc_top3 = train_one_epoch(model, train_loader, optimizer_phase1, criterion, device)
        val_loss, val_acc_top1, val_acc_top2, val_acc_top3 = val_one_epoch(model, val_loader, criterion, device)
        scheduler_phase1.step()
        print(f"Phase1 Epoch [{epoch}/{phase1_epochs}] Train Loss: {train_loss:.4f}, Train Acc Top1: {train_acc_top1:.4f}, Top2: {train_acc_top2:.4f}, Top3: {train_acc_top3:.4f} | Val Loss: {val_loss:.4f}, Val Acc Top1: {val_acc_top1:.4f}, Top2: {val_acc_top2:.4f}, Top3: {val_acc_top3:.4f}")
        early_stopping_phase1(val_loss, model)
        if early_stopping_phase1.early_stop:
            print("Early stopping Phase 1")
            break
        if val_acc_top1 > best_acc_phase1:
            best_acc_phase1 = val_acc_top1

    for param in model.parameters():
        param.requires_grad = True
    optimizer_phase2 = torch.optim.AdamW(model.parameters(), lr=lr_phase2, weight_decay=weight_decay)
    scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=phase2_epochs)

    early_stopping_phase2 = EarlyStopping(patience=early_stopping_patience_phase2, verbose=True, path=f'best_{model_name}_phase2.pt')

    best_acc_phase2 = best_acc_phase1
    print("\n========== Phase 2: Fine-tune all layers ==========")
    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_acc_top1, train_acc_top2, train_acc_top3 = train_one_epoch(model, train_loader, optimizer_phase2, criterion, device)
        val_loss, val_acc_top1, val_acc_top2, val_acc_top3 = val_one_epoch(model, val_loader, criterion, device)
        scheduler_phase2.step()
        print(f"Phase2 Epoch [{epoch}/{phase2_epochs}] Train Loss: {train_loss:.4f}, Train Acc Top1: {train_acc_top1:.4f}, Top2: {train_acc_top2:.4f}, Top3: {train_acc_top3:.4f} | Val Loss: {val_loss:.4f}, Val Acc Top1: {val_acc_top1:.4f}, Top2: {val_acc_top2:.4f}, Top3: {val_acc_top3:.4f}")
        early_stopping_phase2(val_loss, model)
        if early_stopping_phase2.early_stop:
            print("Early stopping Phase 2")
            break
        if val_acc_top1 > best_acc_phase2:
            best_acc_phase2 = val_acc_top1

    print(f"\nTraining complete. Best val acc after Phase 2 = {best_acc_phase2:.4f}")

if __name__ == "__main__":
    main(
        model_name='cbam_resnet50',
        pretrained=True,
        dropout_rate=0.5,
        optimizer_type='AdamW',
        lr_phase1=1e-4,
        lr_phase2=1e-5,
        weight_decay=1e-4,
        batch_size=8,
        augment_level='medium',
        label_smoothing_epsilon=0.1,
        early_stopping_patience_phase1=80,
        early_stopping_patience_phase2=200
    )
