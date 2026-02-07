import os
from pathlib import Path

# ---- MNIST & CIFAR10 (torchvision) ----
from torchvision import datasets

# ---- MedMNIST (OrganAMNIST) ----
import medmnist
from medmnist import INFO

# ----------------- Helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------- MNIST -----------------
def download_mnist(download_root: Path = Path("./data/MNIST")):
    print("==> Downloading MNIST")
    ensure_dir(download_root)
    datasets.MNIST(root=str(download_root), train=True, download=True)
    datasets.MNIST(root=str(download_root), train=False, download=True)
    print(f"[OK] MNIST downloaded to: {download_root}")

# ----------------- CIFAR10 -----------------
def download_cifar10(download_root: Path = Path("./data/CIFAR10")):
    print("==> Downloading CIFAR-10")
    ensure_dir(download_root)
    datasets.CIFAR10(root=str(download_root), train=True, download=True)
    datasets.CIFAR10(root=str(download_root), train=False, download=True)
    print(f"[OK] CIFAR-10 downloaded to: {download_root}")

# ----------------- OrganAMNIST (MedMNIST) -----------------
def download_organamnist(download_root: Path = Path("./data/OrganAMNIST")):
    print("==> Downloading OrganAMNIST (MedMNIST)")
    ensure_dir(download_root)
    data_flag = "organamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    DataClass(split="train", download=True, root=str(download_root))
    DataClass(split="val", download=True, root=str(download_root))
    DataClass(split="test", download=True, root=str(download_root))
    print(f"[OK] OrganAMNIST downloaded to: {download_root}")

# ----------------- Main -----------------
if __name__ == "__main__":
    download_mnist()
    download_cifar10()
    download_organamnist()

    print("\nDone. All datasets downloaded to ./data/{MNIST,CIFAR10,OrganAMNIST}/")
