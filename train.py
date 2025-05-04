import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from timm import create_model
from torch.amp import autocast, GradScaler

# ✅ Configuration
DATA_DIR = r'D:\Face Recognition-Dip Project\train'
CHECKPOINT_DIR = r'D:\Face Recognition-Dip Project\checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# ✅ Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using Device: {device}")

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load Dataset
try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"✅ Dataset Loaded: {len(dataset)} images | {num_classes} classes")
except Exception as e:
    print(f"❌ Dataset Load Error: {e}")
    sys.exit()

# ✅ Train Function
def train():
    # ✅ DataLoader
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )

    # ✅ Model Setup
    try:
        model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(NUM_EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # ✅ Progress Log (Optional)
            sys.stdout.write(
                f"\r🔄 Batch {batch_idx}/{len(train_loader)} - "
                f"Loss: {loss.item():.4f} | Accuracy: {(correct / total * 100):.2f}%"
            )
            sys.stdout.flush()

        avg_loss = total_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"\n✅ Epoch {epoch + 1} Complete - Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # 💾 Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vit_vggface2_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    # 💾 Save final model
    final_path = os.path.join(CHECKPOINT_DIR, 'vit_vggface2_final.pth')
    torch.save(model.state_dict(), final_path)
    print("✅ Training Complete! Final model saved.")

# ✅ Windows-safe start
if __name__ == '__main__':
    train()
