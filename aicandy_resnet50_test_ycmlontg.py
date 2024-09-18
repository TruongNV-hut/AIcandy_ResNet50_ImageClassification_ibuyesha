import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from aicandy_model_src_supduior.aicandy_resnet50_model_ycrrignm import resnet50

# python aicandy_resnet50_test_ycmlontg.py --image_path ../image_test.jpg --model_path aicandy_model_out_lgqllayc/aicandy_model_pth_ydvnemld.pth --label_path label.txt


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    print('labels: ',labels)
    return labels

def predict(image_path, model_path, label_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(label_path)
    num_classes = len(labels)
    
    # Khởi tạo mô hình và tải trọng số
    model = resnet50(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = predicted.item()
    return labels.get(predicted_class, "Unknown")

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label file')

    args = parser.parse_args()
    predicted_class = predict(args.image_path, args.model_path, args.label_path)
    print(f'Predicted class: {predicted_class}')
