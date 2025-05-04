import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from timm import create_model
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def evaluate():
    # ✅ Configuration
    DATASET_PATH = r"D:\Face Recognition-Dip Project\train"  # Test folder path (structured like train)
    MODEL_PATH = r"D:\Face Recognition-Dip Project\checkpoints\vit_vggface2_final.pth"  # Path to trained model
    MODEL_NAME = 'vit_base_patch16_224'  # Should match training
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    # ✅ Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using Device: {DEVICE}")

    # ✅ Check model path
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        return

    # ✅ Data transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ✅ Load dataset
    try:
        test_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )
        class_names = test_dataset.classes
        print(f"✅ Loaded {len(test_dataset)} test images across {len(class_names)} classes.")
    except Exception as e:
        print(f"❌ Dataset Load Error: {e}")
        return

    # ✅ Load model
    try:
        model = create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return

    # ✅ Evaluation
    all_preds, all_labels = [], []
    print("🔍 Starting Evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ✅ Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

    # ✅ Detailed Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("📊 Classification Report:")
    print(report)

# ✅ Windows-safe entry point
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    evaluate()
