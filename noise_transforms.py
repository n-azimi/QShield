import torch
import random
from torchvision import transforms

# Define geometric noise transform
class AddGeometricNoise(object):
    def __init__(self, dataset_name):
        if dataset_name == 'MNIST':
            self.size = (28, 28)  # MNIST image size
        elif dataset_name == 'CIFAR10':
            self.size = (32, 32)  # CIFAR-10 image size
        else:
            raise ValueError("Unsupported dataset type. Choose 'MNIST' or 'CIFAR10'.")

        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),                               # Random rotation by ±10 degrees
            transforms.RandomResizedCrop(size=self.size, scale=(0.95, 1.0)),     # Slight zoom/crop
            # transforms.RandomHorizontalFlip(p=0.1),                            # Reduced flip probability
            transforms.RandomAffine(degrees=0, translate=(0.01, 0.01))           # Small translation
        ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

# Define Gaussian noise transform
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# Define Salt and Pepper noise transform
class AddSaltPepperNoise(object):
    def __init__(self, prob=0.01):
        """
        prob: Probability of each pixel being altered by noise.
        """
        self.prob = prob

    def __call__(self, tensor):
        # Get the number of pixels to alter
        num_salt = int(self.prob * tensor.numel() * 0.5)    # Half for salt
        num_pepper = int(self.prob * tensor.numel() * 0.5)  # Half for pepper

        # Flatten the tensor for easy indexing
        flat_tensor = tensor.view(-1)

        # Randomly choose indices for salt and pepper noise
        salt_indices = torch.randperm(flat_tensor.size(0))[:num_salt]
        pepper_indices = torch.randperm(flat_tensor.size(0))[:num_pepper]

        # Apply salt (set to 1)
        flat_tensor[salt_indices] = 1.0

        # Apply pepper (set to 0)
        flat_tensor[pepper_indices] = 0.0

        # Reshape back to original dimensions
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"

# Define Occlusion noise transform
class AddOcclusionNoise(object):
    def __init__(self, occlusion_size=8):
        """
        occlusion_size: The size of the square occlusion box.
        """
        self.occlusion_size = occlusion_size

    def __call__(self, tensor):
        _, h, w = tensor.size()  # Get the dimensions of the image (C, H, W)

        # Randomly choose the top-left corner of the occlusion box
        top = random.randint(0, h - self.occlusion_size)
        left = random.randint(0, w - self.occlusion_size)

        # Set the pixels in the box to 0 (black box)
        # tensor[:, top:top + self.occlusion_size, left:left + self.occlusion_size] = 0.0

        # Set the pixels in the box to 1 (White box)
        tensor[:, top:top + self.occlusion_size, left:left + self.occlusion_size] = 1.0

        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(occlusion_size={self.occlusion_size})"

# Define Poisson noise transform
class AddPoissonNoise(object):
    def __init__(self, scale=1.0):
        """
        scale: Scaling factor to control the level of Poisson noise.
        """
        self.scale = scale

    def __call__(self, tensor):
        # Apply Poisson noise to the image
        # Scale the pixel values to increase the effect of Poisson noise
        noisy_tensor = torch.poisson(tensor * self.scale) / self.scale
        return noisy_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale})"
