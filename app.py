import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image
import pennylane as qml
from pennylane import numpy as np
from qnn_toolkit import QNN, FeatureExtractor, DynamicWeightingModule

# ==========================================================================================
# DEVICE CONFIGURATION
# ==========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================================
# MODEL ARCHITECTURES
# ==========================================================================================

class CNN_Grayscale(nn.Module):
    """ResNet18-based CNN for grayscale images (MNIST, OrganAMNIST)."""
    
    def __init__(self, num_classes=10):
        super(CNN_Grayscale, self).__init__()
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet18 = models.resnet18(weights=weights)
        
        # Adapt first conv layer: RGB (3 channels) -> Grayscale (1 channel)
        old_conv = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet18.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        
        # Adapt output layer
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.resnet18(x)
        return self.logsoftmax(x)


class CNN_CIFAR10(nn.Module):
    """ResNet18-based CNN for color images (CIFAR-10)."""
    
    def __init__(self, num_classes=10):
        super(CNN_CIFAR10, self).__init__()
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet18 = models.resnet18(weights=weights)
        
        # Adapt output layer
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.resnet18(x)
        return self.logsoftmax(x)


class DNN_Grayscale(nn.Module):
    """Fully connected DNN for grayscale images (MNIST, OrganAMNIST)."""
    
    def __init__(self, num_classes=10):
        super(DNN_Grayscale, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.2)
        
        # Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        
        # Layer 3
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class DNN_CIFAR10(nn.Module):
    """Fully connected DNN for color images (CIFAR-10)."""
    
    def __init__(self, num_classes=10):
        super(DNN_CIFAR10, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.2)
        
        # Layer 2
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.2)
        
        # Layer 3
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.2)
        
        # Layer 4
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(0.2)
        
        # Output layer
        self.fc5 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.drop4(x)
        
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# ==========================================================================================
# MODEL LOADING UTILITIES
# ==========================================================================================

def load_model(model_path, model_class, *args, **kwargs):
    """
    Load a classical model (CNN/DNN) from saved checkpoint.
    
    Args:
        model_path: Path to saved model file
        model_class: Model class to instantiate
        *args, **kwargs: Arguments for model class constructor
    
    Returns:
        Loaded model in eval mode
    """
    # Create model instance
    instance = model_class(*args, **kwargs)
    
    # Set global aliases for unpickling (saved models expect 'CNN'/'DNN' names)
    if "cnn_model" in model_path:
        globals()['CNN'] = model_class
    elif "dnn_model" in model_path:
        globals()['DNN'] = model_class
    
    # Load saved model and extract state dict
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
    instance.load_state_dict(loaded_model.state_dict())
    
    # Move to device and set to eval mode
    instance.to(device)
    instance.eval()
    
    return instance


def load_qnn_model(model_path):
    """
    Load a Quantum Neural Network model from saved checkpoint.
    
    Args:
        model_path: Path to saved QNN model file
    
    Returns:
        Loaded QNN model in eval mode
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Set internal modules to eval mode
    if hasattr(model, 'dynamic_weight_module') and model.dynamic_weight_module is not None:
        model.dynamic_weight_module.eval()
    
    return model

# ==========================================================================================
# LOAD ALL MODELS
# ==========================================================================================

models_dict = {
    'mnist': {
        'cnn': load_model('clean_models/clean_model-MNIST-cnn_model.pt', CNN_Grayscale, num_classes=10),
        'dnn': load_model('clean_models/clean_model-MNIST-dnn_model.pt', DNN_Grayscale, num_classes=10),
        'qnn': load_qnn_model('clean_models/clean_model-MNIST-qnn_model-full_entanglement_ansatz.pt')
    },
    'organmnist': {
        'cnn': load_model('clean_models/clean_model-OrganAMNIST-cnn_model.pt', CNN_Grayscale, num_classes=11),
        'dnn': load_model('clean_models/clean_model-OrganAMNIST-dnn_model.pt', DNN_Grayscale, num_classes=11),
        'qnn': load_qnn_model('clean_models/clean_model-OrganAMNIST-qnn_model-full_entanglement_ansatz.pt')
    },
    'cifar10': {
        'cnn': load_model('clean_models/clean_model-CIFAR10-cnn_model.pt', CNN_CIFAR10, num_classes=10),
        'dnn': load_model('clean_models/clean_model-CIFAR10-dnn_model.pt', DNN_CIFAR10, num_classes=10),
        'qnn': load_qnn_model('clean_models/clean_model-CIFAR10-qnn_model-full_entanglement_ansatz.pt')
    }
}

# ==========================================================================================
# DATASET CONFIGURATION
# ==========================================================================================

# Class labels for each dataset
ORGANMNIST_LABELS = {
    0: 'Bladder', 1: 'Femur-left', 2: 'Femur-right', 3: 'Heart', 4: 'Kidney-left',
    5: 'Kidney-right', 6: 'Liver', 7: 'Lung-left', 8: 'Lung-right', 9: 'Pancreas', 10: 'Spleen'
}

CIFAR10_LABELS = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# ==========================================================================================
# IMAGE PREPROCESSING
# ==========================================================================================

def transform_image(image_bytes, dataset='mnist'):
    """
    Transform uploaded image for model inference.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        dataset: Dataset name ('mnist', 'organmnist', or 'cifar10')
    
    Returns:
        Transformed tensor with shape (1, C, H, W)
    """
    dataset = dataset.lower()
    
    if dataset == 'cifar10':
        # CIFAR-10: RGB images, 32x32
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ])
    
    elif dataset == 'mnist':
        # MNIST: Grayscale images, 28x28
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
    elif dataset == 'organmnist':
        # OrganAMNIST: Grayscale images, 28x28
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Apply transformation and add batch dimension
    return transform(image).unsqueeze(0)

# ==========================================================================================
# FLASK APPLICATION
# ==========================================================================================

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image classification prediction request.
    
    Expected form data:
        - file: Image file
        - dataset: Dataset name (mnist/organmnist/cifar10)
        - model_type: Model type (cnn/dnn/qnn)
    
    Returns:
        JSON response with prediction result or error
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    dataset = request.form.get('dataset', 'mnist')
    model_type = request.form.get('model_type', 'cnn')
    
    try:
        # Get model
        model = models_dict[dataset][model_type]
        
        # Preprocess image
        img_bytes = file.read()
        tensor = transform_image(img_bytes, dataset).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
        
        # Get prediction
        _, predicted = torch.max(outputs.data, 1)
        prediction_idx = predicted.item()
        
        # Map index to label
        if dataset == 'organmnist':
            prediction = ORGANMNIST_LABELS.get(prediction_idx, 'Unknown')
        elif dataset == 'cifar10':
            prediction = CIFAR10_LABELS.get(prediction_idx, 'Unknown')
        else:  # mnist
            prediction = prediction_idx
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================================================================
# APPLICATION ENTRY POINT
# ==========================================================================================

if __name__ == '__main__':
    app.run(debug=False)
