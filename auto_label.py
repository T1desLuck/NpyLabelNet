import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import glob
import argparse
import json
from tqdm import tqdm  # Добавляем tqdm


class MiniCNN(nn.Module):
    def __init__(self, num_classes=100):  # Изменено на 100 классов
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_classes(json_path):
    try:
        with open(json_path, 'r') as f:
            classes = json.load(f)
        return classes
    except Exception as e:
        raise ValueError(f"Error loading classes.json: {e}")


def preprocess_image(image):
    try:
        image = image.convert("RGB")
        img_array = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
        return img_tensor
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")


def main():
    parser = argparse.ArgumentParser(description="NpyLabelNet: Generate .npy labels")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, required=True, help="Path to output labels")
    parser.add_argument("--model_path", type=str, default="models/tiny_cnn.pth", help="Path to model weights")
    parser.add_argument("--classes_path", type=str, default="classes.json", help="Path to classes.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MiniCNN(num_classes=100).to(device)  # Изменено на 100 классов
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model weights not found at {args.model_path}. Train the model using train.py.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()

    # Load classes
    classes = load_classes(args.classes_path)

    # Check input/output paths
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        return
    os.makedirs(args.output, exist_ok=True)

    # Process images
    image_paths = sorted(glob.glob(os.path.join(args.input, "*.[jp][pn]g")))
    if not image_paths:
        print(f"No images found in {args.input}")
        return

    for img_path in tqdm(image_paths, desc="Processing images"):  # Добавлен tqdm
        img_name = os.path.basename(img_path)
        npy_path = os.path.join(args.output, img_name.replace(".jpg", ".npy").replace(".png", ".npy"))
        if os.path.exists(npy_path):
            print(f"Skipping {img_name}: .npy already exists")
            continue

        try:
            image = Image.open(img_path)
            if image.size != (512, 512):
                print(f"Skipping {img_name}: size {image.size}, expected (512, 512)")
                continue
            input_tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                max_prob, class_id = torch.max(probabilities, dim=1)
                class_id = class_id.item()
                max_prob = max_prob.item()
                if max_prob < 0.5:
                    class_id = 99  # Изменено на 99 для "неопределённый_объект"
                class_name = classes[str(class_id)]["name"]
                np.save(npy_path, np.array([class_id], dtype=np.int64))
                print(f"Processed {img_name}: class {class_id} ({class_name}), confidence {max_prob:.2f}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue


if __name__ == "__main__":
    main()
