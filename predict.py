import os
import torch
from torchvision import transforms
from PIL import Image
from timm import create_model

# ✅ Configuration
MODEL_PATH = r"D:\Face Recognition-Dip Project\checkpoints\vit_vggface2_final.pth"
CLASS_DIR = r"D:\Face Recognition-Dip Project\train"  # Same directory used in training
IMG_PATH = r"D:\Face Recognition-Dip Project\train\n000389\0034_01.jpg"  # Replace with your image

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Class Names
class_names = sorted(os.listdir(CLASS_DIR))
num_classes = len(class_names)

# ✅ Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load and Preprocess Image
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)

# ✅ Load Model
def load_model():
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ✅ Predict Function
def predict(img_path):
    image = load_image(img_path)
    model = load_model()

    with torch.no_grad():
        outputs = model(image)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = class_names[predicted_idx]

    print(f"✅ Predicted Class: {predicted_class} (Folder Name)")

# ✅ Run
if __name__ == '__main__':
    predict(IMG_PATH)
