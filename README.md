# QShield: Securing Neural Networks Against Adversarial Attacks Using Quantum Circuits


[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)](#)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20MacOS%20%7C%20Windows-lightgrey.svg)](#)
[![No Maintenance Intended](https://img.shields.io/badge/Status-Maintained-green.svg)](#)



## 📚 Table of Contents

1. [📋 Overview](#-overview)
2. [💡 Key Features](#-key-features)
3. [🏗️ Architecture & Components](#%EF%B8%8F-architecture--components)
4. [⚛️ Parameterized Quantum Circuits](#%EF%B8%8F-parameterized-quantum-circuits)
5. [🛢️ Datasets](#%EF%B8%8F-datasets)
6. [🧠 Models](#-models)
7. [⚙️ Installation & Setup](#%EF%B8%8F-installation--setup)
8. [🗂️ Code Structure](#%EF%B8%8F-code-structure)
9. [📦 Libraries & Dependencies](#-libraries--dependencies)
10. [📑 Citation](#-citation)
11. [📜 License](#-license)



## 📋 Overview

In this work, we introduce QShield, a modular hybrid quantum–classical neural network (HQCNN) architecture designed to enhance the adversarial robustness of classical deep learning models. QShield integrates a conventional convolutional neural network (CNN) backbone for feature extraction with a quantum processing module that encodes the extracted features into quantum states, applies structured entanglement operations under realistic noise models, and outputs a hybrid prediction through a dynamically weighted fusion mechanism implemented via a lightweight multilayer perceptron (MLP). We systematically evaluate both classical and hybrid quantum–classical models on the MNIST, OrganAMNIST, and CIFAR-10 datasets, using a comprehensive set of robustness, efficiency, and computational performance metrics.

Our results demonstrate that classical models are highly vulnerable to adversarial attacks, whereas the proposed hybrid models with entanglement patterns maintain high predictive accuracy while substantially reducing attack success rates across a wide range of adversarial attacks. Across all evaluated datasets, the hybrid models consistently outperformed CNN baselines, achieving robustness gains ranging from modest improvements against gradient-based attacks to over an order-of-magnitude reduction in attack success rates for optimization- and query-based attacks. Furthermore, the proposed hybrid architecture significantly increased the computational cost required to generate adversarial examples, thereby introducing an additional layer of defense. These findings indicate that the proposed modular hybrid architecture achieves a practical balance between predictive accuracy and adversarial robustness, positioning it as a promising approach for secure and reliable machine learning in sensitive and safety-critical applications.



## 💡 Key Features

* **QShield Architecture**: A modular hybrid quantum–classical pipeline that combines CNN-based feature extraction with parameterized quantum circuits for robust prediction.
* **Device-Aware Training**: Automatic detection of available hardware with seamless CUDA/CPU device placement for efficient training and evaluation.
* **Jupyter Notebook Integration**: Ready-to-run training and evaluation pipelines with integrated logging, dataset selection, and adversarial attack configuration.
* **Configurable Settings**: Flexible parameterization of entanglement depth, encoding strategy, noise strength, number of output classes, and optimizer/loss functions.
* **Noise Modeling**: Built-in simulation of realistic quantum noise processes (depolarizing, amplitude damping, phase damping, and mixed noise), with optional input noise injection.
* **Various Qubit Entanglement Patterns**: Support for multiple qubit entanglement patterns, including none, linear, star, and fully connected configurations.
* **Adaptive Hybrid Fusion**: An MLP-based dynamic fusion mechanism that adaptively balances quantum and classical predictions on a per-input basis.
* **Flexible Encoding Methods**: Support for both standard angle encoding and enhanced multi-gate RX/RY/RZ encoding, with PCA or orthogonal expansion for dimensionality alignment.
* **Robustness Evaluation**: Systematic benchmarking under diverse adversarial attacks (FGSM, PGD, DeepFool, C&W, Square, etc.) across MNIST, CIFAR-10, and OrganAMNIST datasets.



## 🏗️ Architecture & Components

<p align="center">
  <img src="/img/_DNN.png" alt="Fully connected DNN architectures for MNIST, OrganAMNIST, and CIFAR-10 datasets" width="500">
  <br>
  Figure 1. Fully connected DNN architectures for MNIST, OrganAMNIST, and CIFAR-10 datasets
</p>

<br>

<p align="center">
  <img src="/img/_CNN.png" alt="CNN architectures based on the ResNet-18 backbone for MNIST, OrganAMNIST, and CIFAR-10 datasets" width="500">
  <br>
  Figure 2. CNN architectures based on the ResNet-18 backbone for MNIST, OrganAMNIST, and CIFAR-10 datasets
</p>

<br>

<p align="center">
  <img src="/img/_QShield.png" alt="Schematic overview of the proposed QShield architecture" width="500">
  <br>
  Figure 3. Schematic overview of the proposed QShield architecture
</p>



## ⚛️ Parameterized Quantum Circuits

<p align="center">
  <img src="/img/No-Ent.png" alt="No entanglement quantum circuit" width="500">
  <br>
  Figure 4. No entanglement quantum circuit
</p>

<br>

<p align="center">
  <img src="/img/Linear-Ent.png" alt="Linear entanglement quantum circuit" width="500">
  <br>
  Figure 5. Linear entanglement quantum circuit
</p>

<br>

<p align="center">
  <img src="/img/Star-Ent.png" alt="Star entanglement quantum circuit" width="500">
  <br>
  Figure 6. Star entanglement quantum circuit
</p>

<br>

<p align="center">
  <img src="/img/Full-Ent.png" alt="Full entanglement quantum circuit" width="500">
  <br>
  Figure 7. Full entanglement quantum circuit
</p>



## 🛢️ Datasets

QShield experiments are evaluated on widely used benchmark datasets for both general-purpose and medical imaging tasks:

* **📝 [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)** 
  Handwritten digit classification dataset containing 70,000 grayscale images (28×28) across 10 classes (digits 0–9).

* **🖼️ [CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)**
  Natural image dataset with 60,000 color images (32×32) across 10 object classes (airplane, car, cat, dog, etc.).

* **🫀 [OrganAMNIST](https://medmnist.com/)**
  A medical imaging benchmark derived from abdominal CT scans, containing 11 organ classes for classification.

All datasets used in QShield can also be accessed directly via the following Hugging Face repository:

🔗 [QShield Datasets on Hugging Face](https://huggingface.co/datasets/QShield-hf/Dataset/tree/main)



## 🧠 Models

The trained DNN, CNN, and QNN models are publicly available and can be accessed via the following Hugging Face link:

🔗 [QShield Models on Hugging Face](https://huggingface.co/QShield-hf/Model/tree/main)



## ⚙️ Installation & Setup

#### 1. Environment Setup
```bash
#🔹Create a virtual environment (recommended)
python -m venv qshield

#🔹Activate the environment

# On Linux/macOS:
source qshield/bin/activate

# On Windows:
qshield\Scripts\activate

# If using Conda:
conda activate qshield
```
```
# Install dependencies
pip install -r requirements.txt
```


#### 2. Parameters Configuration

This section lists the key options and parameters you can configure before training and evaluation.


##### Device Selection

```python
global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

##### Model & Dataset Settings

```python
# Choose the quantum circuit pattern:
# Options: no_entanglement_ansatz, linear_entanglement_ansatz, full_entanglement_ansatz, star_entanglement_ansatz
global entanglement_type
entanglement_type = 'no_entanglement_ansatz'

# Choose the dataset:
# Options: MNIST, CIFAR10, OrganAMNIST
global dataset_name
dataset_name = 'OrganAMNIST'

# Choose the base neural network:
# Options: CNN-MNIST, DNN-MNIST, CNN-CIFAR10, DNN-CIFAR10, CNN-OrganAMNIST, DNN-OrganAMNIST
global NN_name
NN_name = 'DNN-OrganAMNIST'
```


##### Adversarial Attack Settings

```python
# Adversarial attack method:
# Options: fgsm_attack, pgd_attack, apgd_attack, vnifgsm_attack, vmifgsm_attack, 
#          sinifgsm_attack, cw_attack, deepfool_attack, onepixel_attack, square_attack
global adversarial_attack_name
adversarial_attack_name = 'fgsm_attack'
```


##### Training Parameters

```python
# Number of training epochs
global num_epochs
num_epochs = 10

# Number of classes:
# 10 → MNIST / CIFAR10
# 11 → OrganAMNIST
global NUM_CLASSES
NUM_CLASSES = 11
```


##### Quantum Neural Network (QNN) Settings

```python
# Enable / disable QNN model construction
global construct_qnn_model
construct_qnn_model = True

# QNN configuration
qnn_settings = {
    "output_dim": NUM_CLASSES,                       # Number of output classes
    "circuit_depth": 1,                              # Depth of the quantum circuit
    "noise_strength": 0.005,                         # Initial quantum noise strength (0.0 – 1.0)
    "input_noise_injection": False,                  # Apply random perturbations to inputs
    "entanglement_type": 'no_entanglement_ansatz',   # Entanglement type for the circuit
    "use_dynamic_weights": True,                     # Dynamic weighting for hybrid fusion
    "encoding_method": 'enhanced_angle',             # Data encoding method: 'angle' or 'enhanced_angle'
    "noise_model": 'mixed'                           # Noise model: 'depolarizing', 'amplitude_damping',
                                                     #              'phase_damping', or 'mixed'
}
```


##### Target Model, Optimizer & Loss

```python
# Select the target model:
# Options: dnn_model, cnn_model, qnn_model
global target_model
target_model = dnn_model

# Optimizer
optimizer = optim.Adam(target_model.parameters())

# Loss function
loss_fn = nn.NLLLoss()
```



## 🗂️ Code Structure

```
📁 QShield/
├── 🧠 clean_models/                                             # Models
│   └── clean_model-<dataset>-<model_type>.pt
│   └── ...
├── 🛢️ data/                                                     # Datasets
│   └── CIFAR10/
│   └── MNIST/
│   └── OrganAMNIST/
├── 📚 docs/                                                     # Documentations
│   └── Adversarial Attack Parameters.md
│   └── Dataset Visualization.ipynb
│   └── Results.ipynb
├── 🖼️ img/                                                      # Figures and architecture diagrams
│   └── _DNN.png
│   └── _CNN.png
│   └── _QShield.png
├── ➕ etc/                                                      # Integrated Jupyter Notebooks
│   └── Jupyter Notebook - Adversarial Attacks.ipynb
│   └── Jupyter Notebook - Model Training & Evaluation.ipynb
├── 📓 Jupyter Notebook.ipynb                                    # Primary Jupyter Notebook
├── 📄 LICENSE                                                   # License
├── 🌀 noise_transforms.py                                       # Optional Noise Transforms
├── ⚛️ qnn_toolkit.py                                            # Quanvolutional Neural Network Toolkit
├── 📦 requirements.txt                                          # Project Dependencies
└── 📖 README.md                                                 # Primary Documentation
```



## 📦 Libraries & Dependencies

|  Package |  Purpose |
|---------|---------|
| `🔥 torch` | Core deep learning framework (GPU-enabled, PyTorch 2.5.1 + CUDA 12.1) |
| `🖼️ torchvision` | Computer vision utilities, datasets, and transforms |
| `📊 matplotlib` | Visualization and plotting of results |
| `📦 medmnist` | Access to MedMNIST benchmark medical image datasets |
| `🔢 numpy` | Fundamental numerical computing library |
| `📐 scipy` | Scientific computing and optimization tools |
| `🖌 pillow` | Image processing (loading, saving, and transforming images) |
| `🎯 scikit-learn` | Machine learning utilities for metrics, preprocessing, and models |
| `⏳ tqdm` | Progress bars for loops and training |
| `🖥 psutil` | System and resource monitoring (CPU/GPU/memory usage) |
| `🛡️ torchattacks` | PyTorch-based adversarial attack implementations |
| `🛡️ adversarial-robustness-toolbox` | Advanced adversarial attacks and defenses (ART) |
| `⚛️ pennylane` | Quantum machine learning and hybrid quantum-classical circuits |
| `📋 tabulate` | Pretty-print tabular results (ODR, ASR, etc.) |
| `⚙️ pip / setuptools / wheel` | Python packaging and dependency management |



## 📑 Citation

The paper associated with this work is available at **[this link](https://arxiv.org/abs/XXXX.XXXXX)**.

If this repository’s code, data, or results contribute to your research, please acknowledge our work by citing the following paper:

```
@INPROCEEDINGS{CitationKey,
  author    = {},
  title     = {},
  booktitle = {},
  year      = {},
  pages     = {},
  doi       = {}
}
```



## 📜 License

This software is licensed under the GNU General Public License v3.0 (GPLv3). You are free to use, modify, and distribute this software for both personal and commercial purposes, as long as you comply with the terms of the GPLv3 license. This includes preserving the license notice and making the source code of any derivative works available under the same license.






