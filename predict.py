import argparse
import random
import os
from PIL import Image
import torch
from torchvision import models, transforms

def prepare_image(image_file):
    """Prepares an image for model prediction."""
    try:
        image = Image.open(image_file).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_pipeline(image)

def load_model_from_checkpoint(checkpoint_path):
    """Loads a model from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        architecture = checkpoint.get('arch', None)
        if not architecture or architecture not in models.__dict__:
            raise ValueError("Invalid or unsupported model architecture in the checkpoint.")

        model = models.__dict__[architecture](pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        for param in model.parameters():
            param.requires_grad = False
        
        return model

    except Exception as e:
        raise ValueError(f"Error loading checkpoint: {e}")

def make_prediction(image_tensor, trained_model, computation_device, top_k):
    """Generates predictions for an image."""
    trained_model.to(computation_device)
    image_tensor = image_tensor.unsqueeze(0).to(computation_device)

    trained_model.eval()
    with torch.no_grad():
        try:
            outputs = trained_model(image_tensor)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")

    probabilities = torch.exp(outputs)
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    return top_probs.squeeze().tolist(), top_indices.squeeze().tolist()

def select_random_image(directory):
    """Chooses a random image file from a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    valid_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not valid_images:
        raise ValueError("No valid image files found in the directory.")

    return random.choice(valid_images)

def main():
    parser = argparse.ArgumentParser(description="Classify images with a pre-trained model.")
    parser.add_argument('--img_path', type=str, help="Path to the image to classify.")
    parser.add_argument('--img_dir', type=str, help="Directory to randomly select an image from.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Checkpoint file for the model.")
    parser.add_argument('--categories_file', type=str, help="JSON file mapping categories to names.")
    parser.add_argument('--top_classes', type=int, default=5, help="Number of top predictions to display.")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for computations if available.")

    args = parser.parse_args()

    if args.img_path:
        image_file = args.img_path
    elif args.img_dir:
        image_file = select_random_image(args.img_dir)
        print(f"Selected random image: {image_file}")
    else:
        raise ValueError("Either --img_path or --img_dir must be provided.")

  
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        print("Warning: GPU not available. Falling back to CPU.")

    model = load_model_from_checkpoint(args.model_checkpoint)


    image_tensor = prepare_image(image_file)


    top_probabilities, top_indices = make_prediction(image_tensor, model, device, args.top_classes)


    if args.categories_file:
        import json
        try:
            with open(args.categories_file, 'r') as f:
                category_mapping = json.load(f)
            idx_to_class = {v: k for k, v in model.class_to_idx.items()}
            predicted_labels = [category_mapping[idx_to_class[idx]] for idx in top_indices]
        except Exception as e:
            raise ValueError(f"Error loading or processing categories file: {e}")
    else:
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        predicted_labels = [idx_to_class[idx] for idx in top_indices]


    print("Predictions:")
    for prob, label in zip(top_probabilities, predicted_labels):
        print(f"{label}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
