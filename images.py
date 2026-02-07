import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

# ---- MNIST & CIFAR10 (torchvision) ----
from torchvision import datasets
from torchvision.datasets.utils import download_url

# ---- MedMNIST (OrganAMNIST) ----
import medmnist
from medmnist import INFO

# ----------------- Helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize(name: str) -> str:
    """Make safe filenames/dirs."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

def save_n_per_class(images, labels, label_to_name, out_root: Path, n=3, mode="L"):
    """
    images: iterable/array of images (HWC for RGB, HW or HWC for grayscale)
    labels: iterable of ints
    label_to_name: dict int->str (human-readable class name)
    out_root: Path
    n: how many per class
    mode: "L" for grayscale, "RGB" for color
    """
    ensure_dir(out_root)
    counters = defaultdict(int)

    for idx, (img, y) in enumerate(zip(images, labels)):
        y_int = int(y)
        if counters[y_int] >= n:
            continue

        # Convert to PIL.Image
        arr = np.array(img)
        if mode == "L":
            # ensure 2D uint8
            if arr.ndim == 3:
                arr = arr.squeeze()
            arr = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
        else:
            # RGB
            if arr.ndim == 2:  # (H,W) -> (H,W,1) -> RGB
                arr = np.repeat(arr[..., None], 3, axis=2)
            arr = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="RGB")

        # Subfolder per class; filename includes label
        label_name = label_to_name[y_int]
        class_dir = out_root / sanitize(label_name)
        ensure_dir(class_dir)

        # e.g., "cat_01.jpg", "7_02.jpg", "spleen_03.jpg"
        filename = f"{sanitize(label_name)}_{counters[y_int]+1:02d}.jpg"
        pil_img.save(class_dir / filename, format="JPEG", quality=95)
        counters[y_int] += 1

        # Early exit if we already have n for all classes
        if all(v >= n for k, v in counters.items() if k in label_to_name):
            # We can stop once every class present in label_to_name has n.
            break

    # Report which classes we managed to save
    missing = [label_to_name[k] for k in label_to_name if counters[k] < n]
    if missing:
        print(f"[WARN] Not enough samples found for: {missing}")
    else:
        print(f"[OK] Saved {n} images per class to: {out_root}")

# ----------------- MNIST -----------------
def process_mnist(out_root: Path, n=3, download_root: Path = Path("./data/MNIST")):
    print("==> MNIST")
    ensure_dir(download_root)
    ds = datasets.MNIST(root=str(download_root), train=True, download=True)
    images = ds.data.numpy()         # (N, 28, 28) uint8
    labels = ds.targets.numpy()      # (N,)
    label_to_name = {i: str(i) for i in range(10)}
    save_n_per_class(images, labels, label_to_name, out_root / "MNIST", n=n, mode="L")

# ----------------- CIFAR10 -----------------
def process_cifar10(out_root: Path, n=3, download_root: Path = Path("./data/CIFAR10")):
    print("==> CIFAR-10")
    ensure_dir(download_root)
    ds = datasets.CIFAR10(root=str(download_root), train=True, download=True)
    images = np.array(ds.data)       # (N, 32, 32, 3) uint8
    labels = np.array(ds.targets)    # (N,)
    label_to_name = {i: name for i, name in enumerate(ds.classes)}
    save_n_per_class(images, labels, label_to_name, out_root / "CIFAR10", n=n, mode="RGB")

# ----------------- OrganAMNIST (MedMNIST) -----------------
def process_organamnist(out_root: Path, n=3, download_root: Path = Path("./data/OrganAMNIST")):
    print("==> OrganAMNIST (MedMNIST)")
    ensure_dir(download_root)
    data_flag = "organamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", download=True, root=str(download_root))

    # ds.imgs: (N, 28, 28) uint8 for organamnist
    # ds.labels: (N, 1) ints
    images = ds.imgs
    labels = ds.labels.squeeze()

    # Map int label -> human-readable class name from INFO
    # INFO['label'] is like {"0": "spleen", "1": "right kidney", ...}
    label_to_name = {int(k): v for k, v in info["label"].items()}

    save_n_per_class(images, labels, label_to_name, out_root / "OrganAMNIST", n=n, mode="L")

# ----------------- Main -----------------
if __name__ == "__main__":
    output_root = Path("./img")
    ensure_dir(output_root)

    N_PER_CLASS = 3

    process_mnist(output_root, n=N_PER_CLASS)
    process_organamnist(output_root, n=N_PER_CLASS)
    process_cifar10(output_root, n=N_PER_CLASS)

    print("\nDone. Samples saved under ./img/{MNIST,OrganAMNIST,CIFAR10}/<class>/<class>_01.jpg ...")
