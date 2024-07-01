%%writefile predict_keypoints_colab.py

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path, device):
    base_model = mobilenet_v3_small(pretrained=False)
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier[0].in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 198)
    )
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model = base_model.to(device)
    base_model.eval()
    return base_model

def process_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze(0)
    return img, img_transformed

def predict_keypoints(model, img_transformed, device):
    with torch.no_grad():
        outputs = model(img_transformed.to(device)).cpu().detach().numpy()
    return outputs

def modify_keypoints(outputs):
    for i in range(18, len(outputs[0]), 18):
        if outputs[0][i] < 0.2:
            outputs[0][i:] = 0
            break
    return outputs

def save_keypoints(output_dir, image_name, keypoints):
    try:
        keypoints_path = os.path.join(output_dir, 'Keypoints', f"{image_name}.txt")
        x_coords, y_coords = keypoints[::2], keypoints[1::2]
        with open(keypoints_path, 'w') as f:
            for x, y in zip(x_coords, y_coords):
                f.write(f"{x},{y}\n")
        logging.info(f"Keypoints saved for image: {image_name}")
    except Exception as e:
        logging.error(f"Error saving keypoints for image {image_name}: {e}")

def plot_and_save_image(output_dir, image_name, img, keypoints):
    try:
        plt.figure()
        plt.imshow(img)
        x_coords, y_coords = keypoints[::2], keypoints[1::2]
        plt.scatter(x_coords, y_coords, c='r')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'Images', f"{image_name}.png"))
        plt.close()
        logging.info(f"Image saved with keypoints for image: {image_name}")
    except Exception as e:
        logging.error(f"Error saving image with keypoints for image {image_name}: {e}")

def main(model_path, input_folder, device):
    model = load_model(model_path, device)
    target_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    output_folder = 'PredictedKeypoints'
    os.makedirs(os.path.join(output_folder, 'Keypoints'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'Images'), exist_ok=True)

    total_images = 0
    processed_images = 0
    error_images = 0

    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith('.jpg'):
            total_images += 1
            try:
                image_path = os.path.join(input_folder, image_name)
                img, img_transformed = process_image(image_path, transform)

                outputs = predict_keypoints(model, img_transformed, device)
                outputs = modify_keypoints(outputs)

                orig_width, orig_height = img.size
                x_coords = outputs[0][::2] * orig_width
                y_coords = outputs[0][1::2] * orig_height

                keypoints = np.column_stack((x_coords, y_coords)).flatten()

                # Use rsplit to split only from the right
                base_name = image_name.rsplit('.', 1)[0]

                save_keypoints(output_folder, base_name, keypoints)
                plot_and_save_image(output_folder, base_name, img, keypoints)
                
                processed_images += 1
                logging.info(f"Processed image: {image_name}")
            except Exception as e:
                error_images += 1
                logging.error(f"Error processing image {image_name}: {e}")

    logging.info(f"Total images: {total_images}")
    logging.info(f"Successfully processed images: {processed_images}")
    logging.info(f"Images with errors: {error_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict keypoints using a trained model')
    parser.add_argument('--model_path', type=str, help='Path to the trained model (.pth)')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing input images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args.model_path, args.input_folder, device)

