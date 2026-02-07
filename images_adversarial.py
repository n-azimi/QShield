import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets
import torchvision.transforms as transforms
import torchattacks
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import medmnist
from medmnist import INFO

# ==========================================================================================
# CONFIGURATION
# ==========================================================================================

# Select attack: fgsm_attack, pgd_attack, apgd_attack, vmifgsm_attack, 
#                cw_attack, deepfool_attack, onepixel_attack, square_attack
ADVERSARIAL_ATTACK_NAME = 'onepixel_attack'

# Select model type: cnn, dnn, qnn
MODEL_TYPE = 'dnn'

# Number of images per class to save
N_PER_CLASS = 3

# ==========================================================================================
# DEVICE CONFIGURATION
# ==========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    """Load a classical model (CNN/DNN) from saved checkpoint."""
    instance = model_class(*args, **kwargs)
    if "cnn_model" in model_path:
        globals()['CNN'] = model_class
    elif "dnn_model" in model_path:
        globals()['DNN'] = model_class
    
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
    instance.load_state_dict(loaded_model.state_dict())
    instance.to(device)
    instance.eval()
    
    # Set BatchNorm and Dropout to eval mode explicitly
    for module in instance.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            module.eval()
    
    return instance


def load_qnn_model(model_path):
    """Load a Quantum Neural Network model from saved checkpoint."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    if hasattr(model, 'dynamic_weight_module') and model.dynamic_weight_module is not None:
        model.dynamic_weight_module.eval()
    return model

# ==========================================================================================
# ADVERSARIAL ATTACK UTILITIES
# ==========================================================================================

def create_adversarial_attack(model, attack_name, dataset_name, num_classes, model_type):
    """Create adversarial attack instance."""
    
    # Ensure model is in eval mode
    model.eval()
    
    if attack_name == "fgsm_attack":
        return torchattacks.FGSM(model, eps=8/255)
    
    elif attack_name == "pgd_attack":
        return torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    
    elif attack_name == "apgd_attack":
        return torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, 
                                 seed=0, loss='ce', eot_iter=1, rho=0.75, verbose=False)
    
    elif attack_name == "vmifgsm_attack":
        return torchattacks.VMIFGSM(model, eps=8/255, alpha=2/255, steps=10, 
                                    decay=1.0, N=5, beta=3/2)
    
    elif attack_name == "cw_attack":
        # Use ART library for CW attack
        loss_fn = nn.NLLLoss()  # Use NLLLoss since models output log_softmax
        
        # Create a dummy optimizer (required by ART but not used in attack)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        if dataset_name == "MNIST":
            input_shape = (1, 28, 28)
        elif dataset_name == "CIFAR10":
            input_shape = (3, 32, 32)
        elif dataset_name == "OrganAMNIST":
            input_shape = (1, 28, 28)
        
        classifier = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, 
            input_shape=input_shape, nb_classes=num_classes, clip_values=(0, 1)
        )
        return CarliniL2Method(
            classifier=classifier, confidence=0.0, targeted=False, 
            learning_rate=0.05, binary_search_steps=8, max_iter=5, 
            initial_const=0.01, max_halving=5, max_doubling=5, 
            batch_size=1, verbose=False
        )
    
    elif attack_name == "deepfool_attack":
        return torchattacks.DeepFool(model, steps=50, overshoot=0.05)
    
    elif attack_name == "onepixel_attack":
        return torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
    
    elif attack_name == "square_attack":
        return torchattacks.Square(model, norm='Linf', eps=8/255, n_queries=500, 
                                   n_restarts=1, p_init=0.8, loss='margin', 
                                   resc_schedule=True, seed=0, verbose=False)
    
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def generate_adversarial_example(model, attack, image_tensor, label, attack_name):
    """Generate adversarial example for a single image."""
    
    # Ensure model is in eval mode
    model.eval()
    
    # Disable gradients for all parameters during attack
    # (attacks will enable them as needed)
    for param in model.parameters():
        param.requires_grad = False
    
    # Ensure tensor is contiguous
    image_tensor = image_tensor.contiguous().to(device)
    label_tensor = torch.tensor([label]).long().to(device)
    
    if attack_name == "cw_attack":
        # CW attack uses numpy
        try:
            adv_img = attack.generate(image_tensor.cpu().numpy(), label_tensor.cpu().numpy())
            adv_tensor = torch.tensor(adv_img).contiguous().to(device)
        except Exception as e:
            print(f"    Warning: CW attack failed: {e}")
            raise
    else:
        # Torchattacks library
        try:
            # Ensure requires_grad for attack
            image_tensor.requires_grad = True
            adv_tensor = attack(image_tensor, label_tensor)
            # Ensure output is contiguous
            adv_tensor = adv_tensor.contiguous()
            image_tensor.requires_grad = False
        except Exception as e:
            print(f"    Warning: Attack failed: {e}")
            raise
    
    return adv_tensor


def tensor_to_pil(tensor, mode="L"):
    """Convert torch tensor to PIL Image."""
    # Remove batch dimension and move to CPU
    arr = tensor.squeeze(0).cpu().detach().numpy()
    
    # Denormalize to [0, 255]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    
    if mode == "L":
        # Grayscale: (1, H, W) -> (H, W)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return Image.fromarray(arr, mode="L")
    else:
        # RGB: (3, H, W) -> (H, W, 3)
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr, mode="RGB")

# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    """Make safe filenames/dirs."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def image_to_tensor(img, mode="L"):
    """Convert PIL Image or numpy array to torch tensor."""
    arr = np.array(img)
    
    if mode == "L":
        # Grayscale
        if arr.ndim == 3:
            arr = arr.squeeze()
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        # RGB
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        arr = arr.astype(np.float32) / 255.0
        # Ensure contiguous memory layout
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return tensor.contiguous()


def save_n_per_class_with_adversarial(images, labels, label_to_name, out_root: Path, 
                                       model, attack, attack_name, model_type_name,
                                       dataset_name, n=3, mode="L"):
    """
    Save original images and their adversarial examples.
    
    Args:
        images: iterable/array of images
        labels: iterable of ints
        label_to_name: dict int->str
        out_root: Path
        model: loaded model for adversarial attack
        attack: adversarial attack instance
        attack_name: name of attack (for filename)
        model_type_name: name of model type (for filename)
        dataset_name: name of dataset
        n: how many per class
        mode: "L" for grayscale, "RGB" for color
    """
    ensure_dir(out_root)
    counters = defaultdict(int)
    
    print(f"\nGenerating adversarial examples using {attack_name.upper()} on {model_type_name.upper()} model...")

    for idx, (img, y) in enumerate(zip(images, labels)):
        y_int = int(y)
        if counters[y_int] >= n:
            continue

        # Convert to PIL Image
        arr = np.array(img)
        if mode == "L":
            if arr.ndim == 3:
                arr = arr.squeeze()
            arr = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
        else:
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            arr = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="RGB")

        # Setup paths
        label_name = label_to_name[y_int]
        class_dir = out_root / sanitize(label_name)
        ensure_dir(class_dir)
        
        base_filename = f"{sanitize(label_name)}_{counters[y_int]+1:02d}"
        original_filename = f"{base_filename}.jpg"
        
        # Save original image
        pil_img.save(class_dir / original_filename, format="JPEG", quality=100)
        
        # Generate and save adversarial example
        try:
            # Convert to tensor
            img_tensor = image_to_tensor(pil_img, mode=mode)
            
            # Generate adversarial example
            adv_tensor = generate_adversarial_example(
                model, attack, img_tensor, y_int, attack_name
            )
            
            # Convert back to PIL and save
            adv_pil = tensor_to_pil(adv_tensor, mode=mode)
            adv_filename = f"{base_filename}-{attack_name.replace('_attack', '').upper()}-{model_type_name.upper()}.jpg"
            adv_pil.save(class_dir / adv_filename, format="JPEG", quality=100)
            
            print(f" - Saved: {label_name}/{base_filename} (original + adversarial)")
            
        except Exception as e:
            print(f" - Error generating adversarial for {label_name}/{base_filename}: {e}")
            import traceback
            traceback.print_exc()
        
        counters[y_int] += 1

        # Early exit if we have n for all classes
        if all(counters.get(k, 0) >= n for k in label_to_name):
            break

    missing = [label_to_name[k] for k in label_to_name if counters[k] < n]
    if missing:
        print(f"[WARN] Not enough samples found for: {missing}")
    else:
        print(f"[OK] Saved {n} images per class to: {out_root}")

# ==========================================================================================
# DATASET PROCESSING FUNCTIONS
# ==========================================================================================

def process_mnist(out_root: Path, model_dict, attack_name, model_type, n=3, 
                  download_root: Path = Path("./data/MNIST")):
    print("\n" + "="*80)
    print(" * MNIST")
    print("="*80)
    ensure_dir(download_root)
    
    ds = datasets.MNIST(root=str(download_root), train=True, download=True)
    images = ds.data.numpy()
    labels = ds.targets.numpy()
    label_to_name = {i: str(i) for i in range(10)}
    
    model = model_dict['mnist'][model_type]
    attack = create_adversarial_attack(model, attack_name, "MNIST", num_classes=10, model_type=model_type)
    
    save_n_per_class_with_adversarial(
        images, labels, label_to_name, out_root / "MNIST",
        model, attack, attack_name, model_type, "MNIST", n=n, mode="L"
    )


def process_cifar10(out_root: Path, model_dict, attack_name, model_type, n=3,
                    download_root: Path = Path("./data/CIFAR10")):
    print("\n" + "="*80)
    print(" * CIFAR-10")
    print("="*80)
    ensure_dir(download_root)
    
    ds = datasets.CIFAR10(root=str(download_root), train=True, download=True)
    images = np.array(ds.data)
    labels = np.array(ds.targets)
    label_to_name = {i: name for i, name in enumerate(ds.classes)}
    
    model = model_dict['cifar10'][model_type]
    attack = create_adversarial_attack(model, attack_name, "CIFAR10", num_classes=10, model_type=model_type)
    
    save_n_per_class_with_adversarial(
        images, labels, label_to_name, out_root / "CIFAR10",
        model, attack, attack_name, model_type, "CIFAR10", n=n, mode="RGB"
    )


def process_organamnist(out_root: Path, model_dict, attack_name, model_type, n=3,
                        download_root: Path = Path("./data/OrganAMNIST")):
    print("\n" + "="*80)
    print(" * OrganAMNIST")
    print("="*80)
    ensure_dir(download_root)
    
    data_flag = "organamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", download=True, root=str(download_root))

    images = ds.imgs
    labels = ds.labels.squeeze()
    label_to_name = {int(k): v for k, v in info["label"].items()}
    
    model = model_dict['organmnist'][model_type]
    attack = create_adversarial_attack(model, attack_name, "OrganAMNIST", num_classes=11, model_type=model_type)
    
    save_n_per_class_with_adversarial(
        images, labels, label_to_name, out_root / "OrganAMNIST",
        model, attack, attack_name, model_type, "OrganAMNIST", n=n, mode="L"
    )

# ==========================================================================================
# LOAD MODELS
# ==========================================================================================

def load_all_models():
    """Load all models for MNIST, OrganAMNIST, and CIFAR-10."""
    
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models_dict = {}
    
    # Check if model files exist
    model_dir = Path('clean_models')
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print("Please ensure models are in the 'clean_models' directory")
        return None
    
    try:
        # MNIST models
        print("\nLoading MNIST models...")
        models_dict['mnist'] = {}
        models_dict['mnist']['cnn'] = load_model(
            'clean_models/clean_model-MNIST-cnn_model.pt', 
            CNN_Grayscale, num_classes=10
        )
        print("  - CNN loaded")
        
        models_dict['mnist']['dnn'] = load_model(
            'clean_models/clean_model-MNIST-dnn_model.pt', 
            DNN_Grayscale, num_classes=10
        )
        print("  - DNN loaded")
        
        models_dict['mnist']['qnn'] = load_qnn_model(
            'clean_models/clean_model-MNIST-qnn_model-full_entanglement_ansatz.pt'
        )
        print("  - QNN loaded")
        
        # OrganAMNIST models
        print("\nLoading OrganAMNIST models...")
        models_dict['organmnist'] = {}
        models_dict['organmnist']['cnn'] = load_model(
            'clean_models/clean_model-OrganAMNIST-cnn_model.pt', 
            CNN_Grayscale, num_classes=11
        )
        print("  - CNN loaded")
        
        models_dict['organmnist']['dnn'] = load_model(
            'clean_models/clean_model-OrganAMNIST-dnn_model.pt', 
            DNN_Grayscale, num_classes=11
        )
        print("  - DNN loaded")
        
        models_dict['organmnist']['qnn'] = load_qnn_model(
            'clean_models/clean_model-OrganAMNIST-qnn_model-full_entanglement_ansatz.pt'
        )
        print("  - QNN loaded")
        
        # CIFAR-10 models
        print("\nLoading CIFAR-10 models...")
        models_dict['cifar10'] = {}
        models_dict['cifar10']['cnn'] = load_model(
            'clean_models/clean_model-CIFAR10-cnn_model.pt', 
            CNN_CIFAR10, num_classes=10
        )
        print("  - CNN loaded")
        
        models_dict['cifar10']['dnn'] = load_model(
            'clean_models/clean_model-CIFAR10-dnn_model.pt', 
            DNN_CIFAR10, num_classes=10
        )
        print("  - DNN loaded")
        
        models_dict['cifar10']['qnn'] = load_qnn_model(
            'clean_models/clean_model-CIFAR10-qnn_model-full_entanglement_ansatz.pt'
        )
        print("  - QNN loaded")
        
        print("\n[OK] All models loaded successfully.")
        return models_dict
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================================================================
# MAIN
# ==========================================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVERSARIAL IMAGE GENERATION CONFIGURATION")
    print("="*80)
    print(f"Attack: {ADVERSARIAL_ATTACK_NAME}")
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Images per class: {N_PER_CLASS}")
    
    # Load all models
    models_dict = load_all_models()
    
    if models_dict is None:
        print("\n[ERROR] Cannot proceed without models. Exiting.")
        exit(1)
    
    # Setup output directory
    output_root = Path("./img")
    ensure_dir(output_root)
    
    # Process each dataset
    try:
        process_mnist(output_root, models_dict, ADVERSARIAL_ATTACK_NAME, MODEL_TYPE, n=N_PER_CLASS)
    except Exception as e:
        print(f"\n[ERROR] MNIST processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        process_organamnist(output_root, models_dict, ADVERSARIAL_ATTACK_NAME, MODEL_TYPE, n=N_PER_CLASS)
    except Exception as e:
        print(f"\n[ERROR] OrganAMNIST processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        process_cifar10(output_root, models_dict, ADVERSARIAL_ATTACK_NAME, MODEL_TYPE, n=N_PER_CLASS)
    except Exception as e:
        print(f"\n[ERROR] CIFAR10 processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Original and adversarial images saved under:")
    print(f"  ./img/MNIST/<class>/<class>_XX.jpg")
    print(f"  ./img/MNIST/<class>/<class>_XX-{ADVERSARIAL_ATTACK_NAME.replace('_attack', '').upper()}-{MODEL_TYPE.upper()}.jpg")
    print(f"  ./img/OrganAMNIST/<class>/<class>_XX.jpg")
    print(f"  ./img/OrganAMNIST/<class>/<class>_XX-{ADVERSARIAL_ATTACK_NAME.replace('_attack', '').upper()}-{MODEL_TYPE.upper()}.jpg")
    print(f"  ./img/CIFAR10/<class>/<class>_XX.jpg")
    print(f"  ./img/CIFAR10/<class>/<class>_XX-{ADVERSARIAL_ATTACK_NAME.replace('_attack', '').upper()}-{MODEL_TYPE.upper()}.jpg")
    print("="*80 + "\n")
