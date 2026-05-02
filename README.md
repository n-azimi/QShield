# QShield: Securing Neural Networks Against Adversarial Attacks using Quantum Circuits


[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Paper-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/scholar_lookup?arxiv_id=2604.10933)
[![arXiv](https://img.shields.io/badge/arXiv-2604.10933-b31b1b.svg)](https://arxiv.org/abs/2604.10933)
[![Paper Authors](https://img.shields.io/badge/Paper%20Authors-Azimi%20%7C%20Prakash%20%7C%20Wang%20%7C%20Xiong-0d1117?style=flat&logo=readthedocs&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20MacOS%20%7C%20Windows-lightgrey.svg)](#)
[![No Maintenance Intended](https://img.shields.io/badge/Status-Maintained-green.svg)](#)
[![Media Coverage - Machine Brief](https://img.shields.io/badge/Media%20Coverage-Machine%20Brief-2a9d8f?logo=google-news&logoColor=white)](https://www.machinebrief.com/news/quantum-meets-classical-a-new-shield-against-adversarial-att-nw4y)
[![Media Coverage - Quantum Zeitgeist](https://img.shields.io/badge/Media%20Coverage-Quantum%20Zeitgeist-2a9d8f?logo=google-news&logoColor=white)](https://quantumzeitgeist.com/quantum-circuits-adversarial-defence/)


## 📚 Table of Contents

1. [📋 Overview](#-overview)
2. [💡 Key Features](#-key-features)
3. [🏗️ Architecture & Components](#%EF%B8%8F-architecture--components)
4. [⚛️ Parameterized Quantum Circuits](#%EF%B8%8F-parameterized-quantum-circuits)
5. [🛢️ Datasets](#%EF%B8%8F-datasets)
6. [🧠 Models](#-models)
7. [⚙️ Installation & Setup](#%EF%B8%8F-installation--setup)
8. [🌐 Flask Web Application](#-flask-web-application)
9. [🗂️ Code Structure](#%EF%B8%8F-code-structure)
10. [📦 Libraries & Dependencies](#-libraries--dependencies)
11. [📑 Citation](#-citation)
12. [📜 License](#-license)



## 📋 Overview

Deep neural networks remain highly vulnerable to adversarial perturbations, limiting their reliability in security- and safety-critical applications. To address this challenge, we introduce QShield, a modular hybrid quantum–classical neural network (HQCNN) architecture designed to enhance the adversarial robustness of classical deep learning models. QShield integrates a conventional convolutional neural network (CNN) backbone for feature extraction with a quantum processing module that encodes the extracted features into quantum states, applies structured entanglement operations under realistic noise models, and outputs a hybrid prediction through a dynamically weighted fusion mechanism implemented via a lightweight multilayer perceptron (MLP). We systematically evaluate both classical and hybrid quantum–classical models on the MNIST, OrganAMNIST, and CIFAR-10 datasets, using a comprehensive set of robustness, efficiency, and computational performance metrics.

Our results demonstrate that classical models are highly vulnerable to adversarial attacks, whereas the proposed hybrid models with entanglement patterns maintain high predictive accuracy while substantially reducing attack success rates across a wide range of adversarial attacks. Furthermore, the proposed hybrid architecture significantly increased the computational cost required to generate adversarial examples, thereby introducing an additional layer of defense.

These findings indicate that the proposed modular hybrid architecture achieves a practical balance between predictive accuracy and adversarial robustness, positioning it as a promising approach for secure and reliable machine learning in sensitive and safety-critical applications.

⚠️ In the codebase, we use the abbreviation QNN (Quantum Neural Network) instead of HQCNN (Hybrid Quantum–Classical Neural Network). Both terms refer to the same architecture in this work.


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
```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Download datasets
python datasets.py

# (OR) Download datasets + pretrained models from Hugging Face
python hf.py
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



## 🌐 Flask Web Application

```bash
# Download all required datasets
python datasets.py
```

```bash
# Launch the Flask web application
python app.py
```
```bash
# (Optional) Generate additional clean sample images
#    → Edit N_PER_CLASS in images.py, then run:
python images.py
```
```bash
# (Optional) Generate adversarial examples
#    → Open images_adversarial.py and set:
#       ADVERSARIAL_ATTACK_NAME = 'onepixel_attack'   # Options: fgsm_attack, pgd_attack, apgd_attack, vmifgsm_attack,
#                                                     #          cw_attack, deepfool_attack, onepixel_attack, square_attack
#       MODEL_TYPE = 'dnn'                            # Options: cnn, dnn, qnn
#       N_PER_CLASS = 3                               # Images per class to save
#    → Then run:
python images_adversarial.py
```


<p align="center">
  <img src="/img/app/1.png" width="200" style="margin: 10px;">
  <img src="/img/app/2.png" width="200" style="margin: 10px;">
  <img src="/img/app/3.png" width="200" style="margin: 10px;">
  <img src="/img/app/4.png" width="200" style="margin: 10px;">
</p>

<p align="center">
  <img src="/img/app/5.png" width="200" style="margin: 10px;">
  <img src="/img/app/6.png" width="200" style="margin: 10px;">
  <img src="/img/app/7.png" width="200" style="margin: 10px;">
  <img src="/img/app/8.png" width="200" style="margin: 10px;">
</p>


You can select different pretrained HQCNN models by editing the .pt file paths in the LOAD ALL MODELS section of the app.py:

```python
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
```



## 🗂️ Code Structure

```bash
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
├── ➕ etc/                                                      # Integrated Jupyter Notebooks
│   └── Jupyter Notebook - Adversarial Attacks.ipynb
│   └── Jupyter Notebook - Model Training & Evaluation.ipynb
├── 🖼️ img/                                                      # Figures, diagrams, and images
│   └── _DNN.png
│   └── _CNN.png
│   └── _QShield.png
│   └── ...
├── 📃 templates/                                                # HTML structure for QShield webapp
│   └── index.html
├── 🌐 app.py                                                    # Flask web app for image classification using preloaded models
├── 🗃️ datasets.py                                               # Download datasets into local ./data folders
├── ⬇️ hf.py                                                     # Downloads datasets and trained models from Hugging Face
├── 🌄 images.py                                                 # Save sample images from datasets into class-specific folders
├── 🌄 images_adversarial.py                                     # Generate adversarial examples, saving both original and perturbed images per class
├── 📓 Jupyter Notebook.ipynb                                    # Primary Jupyter Notebook
├── 📄 LICENSE                                                   # License
├── 🌀 noise_transforms.py                                       # Optional Noise Transforms
├── ⚛️ qnn_toolkit.py                                            # Quanvolutional Neural Network Toolkit
├── 📖 README.md                                                 # Primary Documentation
└── 📦 requirements.txt                                          # Project Dependencies
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
| `🌐 Flask` | Lightweight web framework for building APIs and web apps |
| `🤗 huggingface_hub` | Download and manage models and datasets from Hugging Face |



## 📑 Citation

The paper associated with this work is available at the links below:

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Paper-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/scholar_lookup?arxiv_id=2604.10933)
[![arXiv](https://img.shields.io/badge/arXiv-2604.10933-b31b1b.svg)](https://arxiv.org/abs/2604.10933)

If this repository or the paper has contributed to your research, please acknowledge our work by citing it:

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Cite-34A853?logo=googlescholar&logoColor=white)](https://scholar.google.com/scholar_lookup?arxiv_id=2604.10933#d=gs_cit&t=1776339950469&u=%2Fscholar%3Fq%3Dinfo%3AdCyx-C3p71cJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)

```
@article{azimi2026qshield,
  title={QShield: Securing Neural Networks Against Adversarial Attacks using Quantum Circuits},
  author={Azimi, Navid and Prakash, Aditya and Wang, Yao and Xiong, Li},
  journal={arXiv preprint arXiv:2604.10933},
  year={2026}
}
```



## 📜 License

This software is licensed under the GNU General Public License v3.0 (GPLv3). You are free to use, modify, and distribute this software for both personal and commercial purposes, as long as you comply with the terms of the GPLv3 license. This includes preserving the license notice and making the source code of any derivative works available under the same license.

