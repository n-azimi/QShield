from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

# ------------------ CONFIG ------------------
MODEL_REPO = "QShield-hf/Model"
DATASET_REPO = "QShield-hf/Dataset"

MODELS_OUT = Path("./clean_models")
DATA_OUT = Path("./data")

# ------------------ HELPERS ------------------
def flatten_folder(parent: Path, inner_name: str):
    inner = parent / inner_name
    if not inner.exists():
        return

    for item in inner.iterdir():
        target = parent / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    inner.rmdir()

# ------------------ DOWNLOAD MODELS ------------------
snapshot_download(
    repo_id=MODEL_REPO,
    allow_patterns="clean_models/**",
    local_dir=MODELS_OUT,
)

print("* Models downloaded.")

# ------------------ DOWNLOAD DATA ------------------
snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    allow_patterns="data/**",
    local_dir=DATA_OUT,
)

print("* Datasets downloaded.")

# ------------------ FLATTEN STRUCTURE ------------------
flatten_folder(MODELS_OUT, "clean_models")
flatten_folder(DATA_OUT, "data")

print("\n* Done! Final structure:")
print("    ./clean_models/*")
print("    ./data/*") 
