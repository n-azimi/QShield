import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Union, List, Optional, Tuple, Dict, Any, Callable, Iterator

class FeatureExtractor(nn.Module):
    """
    Helper class to extract intermediate features from CNN model
    """
    def __init__(self, model: nn.Module, layer_name: str = None):
        """
        Initialize a feature extractor for a CNN model.

        Args:
            model: The CNN model to extract features from
            layer_name: Optional name of the specific layer to extract features from
        """
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate features
        """
        if self.layer_name:
            # If a specific layer name is provided, find and hook that layer
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    module.register_forward_hook(self._hook_fn)
        else:
            # If no layer name is specified, we'll hook the last conv/linear layer
            # before the final classification layer
            last_feature_layer = None
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module != list(self.model.modules())[-1]:
                    last_feature_layer = module

            if last_feature_layer:
                last_feature_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        Hook function to store the features
        """
        self.features = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get both the model output and extract features

        Args:
            x: Input tensor to be processed

        Returns:
            Model's output tensor
        """
        output = self.model(x)
        return output

    def get_features(self) -> torch.Tensor:
        """
        Get the extracted features from the last forward pass

        Returns:
            Extracted feature tensor
        """
        if self.features is None:
            raise ValueError("No features extracted. Run a forward pass first.")
        return self.features

def apply_entanglement_layer(num_qubits: int, entanglement_type: str):
    """
    Apply an entanglement layer based on the selected entanglement type

    Args:
        num_qubits: Number of qubits in the circuit
        entanglement_type: Type of entanglement pattern to apply
    """
    # Define valid entanglement types (TODO: Ring & Grid Topologies)
    valid_types = [
        'no_entanglement_ansatz',
        'linear_entanglement_ansatz',
        'full_entanglement_ansatz',
        'star_entanglement_ansatz'
    ]

    if entanglement_type not in valid_types:
        raise ValueError(f"Invalid entanglement type: '{entanglement_type}'. Valid options are: {', '.join(valid_types)}")

    if entanglement_type == 'no_entanglement_ansatz':
        # No entanglement - do nothing
        pass

    elif entanglement_type == 'linear_entanglement_ansatz':
        # Apply linear entanglement (nearest neighbor)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i+1])

    elif entanglement_type == 'full_entanglement_ansatz':
        # Full entanglement between all qubit pairs
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qml.CNOT(wires=[i, j])

    elif entanglement_type == 'star_entanglement_ansatz':
        # Star topology with first qubit as central node
        central_qubit = 0
        for i in range(1, num_qubits):
            qml.CNOT(wires=[central_qubit, i])

class DynamicWeightingModule(nn.Module):
    """
    Dynamically weight classical and quantum outputs using MLP on statistical features
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 3):
        super(DynamicWeightingModule, self).__init__()

        total_input_dim = input_dim * 2 + 1  # +1 for cross-correlation

        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, classical_output: torch.Tensor, quantum_output: torch.Tensor) -> torch.Tensor:

        c_mean = classical_output.mean(dim=1, keepdim=True)
        q_mean = quantum_output.mean(dim=1, keepdim=True)
        
        # Classical stats
        c_std = classical_output.std(dim=1, keepdim=True)
        c_max = classical_output.max(dim=1, keepdim=True)[0]
        c_kurt = ((classical_output - c_mean)**4).mean(dim=1, keepdim=True) / (c_std**4 + 1e-8)

        # Quantum stats
        q_std = quantum_output.std(dim=1, keepdim=True)
        q_max = quantum_output.max(dim=1, keepdim=True)[0]
        q_kurt = ((quantum_output - q_mean)**4).mean(dim=1, keepdim=True) / (q_std**4 + 1e-8)

        # Cross-correlation
        c_norm = classical_output - c_mean
        q_norm = quantum_output - q_mean
        cross_corr = torch.sum(c_norm * q_norm, dim=1, keepdim=True) / (
            torch.norm(c_norm, dim=1, keepdim=True) *
            torch.norm(q_norm, dim=1, keepdim=True) + 1e-8
        )

        # Concatenate features
        c_features = torch.cat([c_std, c_max, c_kurt], dim=1)
        q_features = torch.cat([q_std, q_max, q_kurt], dim=1)
        combined_features = torch.cat([c_features, q_features, cross_corr], dim=1)  # total 7 features

        # Compute alpha
        alpha = self.network(combined_features)
        return alpha

def angle_encoding_basic(features: torch.Tensor, num_qubits: int):
    """
    Basic angle encoding that maps classical features to quantum rotation angles
    
    Args:
        features: Feature tensor to encode
        num_qubits: Number of qubits to use for encoding
        
    Returns:
        Tuple of (rx_angles, ry_angles, rz_angles) for quantum rotations
        Note: Only ry_angles are used, rx_angles and rz_angles are None
    """
    # Normalize features
    feature_mean = features.mean(dim=1, keepdim=True)
    feature_std = features.std(dim=1, keepdim=True) + 1e-6
    normalized_features = (features - feature_mean) / feature_std
    
    # Create angle features
    cnn_angles = 2 * torch.pi * torch.sigmoid(normalized_features)
    
    # If we have more features than qubits, use feature combination
    if cnn_angles.size(1) > num_qubits:
        # For each qubit, combine multiple features
        features_per_qubit = cnn_angles.size(1) // num_qubits
        combined_angles = []
        
        for i in range(num_qubits):
            start_idx = i * features_per_qubit
            end_idx = min((i + 1) * features_per_qubit, cnn_angles.size(1))
            # Use weighted combination of features for each qubit rotation
            qubit_angles = cnn_angles[:, start_idx:end_idx].mean(dim=1)
            combined_angles.append(qubit_angles)
        
        cnn_angles = torch.stack(combined_angles, dim=1)
    elif cnn_angles.size(1) < num_qubits:
        # If we have fewer features than qubits, duplicate features
        repeats = math.ceil(num_qubits / cnn_angles.size(1))
        cnn_angles = cnn_angles.repeat(1, repeats)[:, :num_qubits]
    
    # Only one type of rotation angle for basic angle encoding
    rx_angles = None
    ry_angles = cnn_angles
    rz_angles = None
    
    return rx_angles, ry_angles, rz_angles

def angle_encoding_enhanced(features: torch.Tensor, num_qubits: int):
    """
    Enhanced angle encoding to better preserve geometric structure
    by using multiple rotation gates per qubit with improved dimensionality handling

    Args:
        features: Feature tensor to encode
        num_qubits: Number of qubits to use for encoding

    Returns:
        Tuple of (rx_angles, ry_angles, rz_angles) for quantum rotations
    """
    batch_size = features.size(0)
    feature_dim = features.size(1)

    # Normalize features
    feature_mean = features.mean(dim=1, keepdim=True)
    feature_std = features.std(dim=1, keepdim=True) + 1e-8  # Epsilon for numerical stability
    normalized_features = (features - feature_mean) / feature_std

    # Calculate number of features we can encode per qubit
    # We'll use 3 rotations per qubit (RX, RY, RZ)
    features_per_qubit = 3
    total_features = num_qubits * features_per_qubit

    # Check total size needed to ensure our reshaping will work
    padded_features_size = batch_size * num_qubits * features_per_qubit

    # Handle the case where we need more features than we have
    if feature_dim < total_features:
        # If we have fewer features than needed:
        # 1. Create a deterministic mapping matrix using orthogonal initialization
        # for better feature separation than random initialization
        if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
            # Use QR decomposition for orthogonal initialization
            mapping_matrix_init = torch.randn(feature_dim, total_features, device=features.device)
            q, r = torch.linalg.qr(mapping_matrix_init)
            # Use the orthogonal component scaled appropriately
            mapping_matrix = q * 0.1
            if mapping_matrix.size(1) < total_features:
                # Pad if needed
                padding = torch.zeros(feature_dim, total_features - mapping_matrix.size(1), device=features.device)
                mapping_matrix = torch.cat([mapping_matrix, padding], dim=1)
        else:
            # Fallback to standard initialization with reduced variance
            # mapping_matrix = torch.randn(feature_dim, total_features, device=features.device) * 0.1
            raise RuntimeError(
            "Feature expansion with orthogonal mapping failed."
            )

        # 2. Project features to higher dimension with gradient preservation
        padded_features = torch.matmul(normalized_features, mapping_matrix)

        # Ensure the padded_features tensor has the right size before reshaping
        if padded_features.numel() != padded_features_size:
            padded_features = padded_features.reshape(batch_size, num_qubits * features_per_qubit)
    else:
        # If we have more features:
        if feature_dim <= total_features * 3:  # If reasonably close, use importance-based selection
            # Use variance as a simple importance metric
            feature_vars = normalized_features.var(dim=0)
            _, indices = torch.topk(feature_vars, total_features)
            padded_features = normalized_features[:, indices]
        else:
            # Use PCA-based approach for dimension reduction
            try:
                # Center the data for PCA
                centered_features = normalized_features - normalized_features.mean(dim=0, keepdim=True)
                
                # Compute covariance matrix
                cov = torch.matmul(centered_features.transpose(0, 1), centered_features) / (batch_size - 1)
                
                # Compute eigendecomposition (PCA)
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    # Sort eigenvectors by eigenvalues in descending order
                    sorted_indices = torch.argsort(eigenvalues, descending=True)
                    eigenvectors = eigenvectors[:, sorted_indices]
                    
                    # Take top principal components
                    principal_components = eigenvectors[:, :total_features]
                    
                    # Project data onto principal components
                    padded_features = torch.matmul(normalized_features, principal_components)

                except RuntimeError as e:
                    raise RuntimeError(
                        "Eigendecomposition for PCA-based approach for dimension reduction failed."
                    ) from e

            except Exception as e:
                print(f"PCA-based approach for dimension reduction failed. Dimensionality reduction error: {e}")

    # Validate the size before reshaping
    if padded_features.numel() != padded_features_size:
        # Create correctly sized tensor and copy data
        temp_features = torch.zeros(batch_size, num_qubits * features_per_qubit, device=features.device)
        min_size = min(temp_features.size(1), padded_features.size(1))
        temp_features[:, :min_size] = padded_features[:, :min_size]
        padded_features = temp_features

    # Reshape
    try:
        reshaped_features = padded_features.view(batch_size, num_qubits, features_per_qubit)
    except RuntimeError:
        # If reshape fails, ensure correct size and try again
        padded_features = torch.zeros(batch_size, num_qubits * features_per_qubit, device=features.device)
        reshaped_features = padded_features.view(batch_size, num_qubits, features_per_qubit)

    # Apply improved scaling for rotation angles
    rx_angles = torch.pi * torch.tanh(reshaped_features[:, :, 0])  # RX: [-π, π]
    ry_angles = torch.pi * torch.tanh(reshaped_features[:, :, 1])  # RY: [-π, π]
    rz_angles = torch.pi * torch.tanh(reshaped_features[:, :, 2])  # RZ: [-π, π]

    return rx_angles, ry_angles, rz_angles

def apply_noise_operations(wires, noise_strength, model_type):
    """
    Apply the selected noise model to the circuit
    """
    if model_type == 'depolarizing':
        # Standard depolarizing channel
        for wire in wires:
            qml.DepolarizingChannel(noise_strength, wires=wire)

    elif model_type == 'amplitude_damping':
        # Amplitude damping
        for wire in wires:
            qml.AmplitudeDamping(noise_strength, wires=wire)

    elif model_type == 'phase_damping':
        # Phase damping
        for wire in wires:
            qml.PhaseDamping(noise_strength, wires=wire)

    elif model_type == 'mixed':
        # Apply a mix of different noise types
        for wire in wires:
            qml.DepolarizingChannel(noise_strength * 0.4, wires=wire)
            qml.AmplitudeDamping(noise_strength * 0.3, wires=wire)
            qml.PhaseDamping(noise_strength * 0.3, wires=wire)

def hybrid_forward(
    input_data: torch.Tensor,
    cnn_model: torch.nn.Module,
    device: torch.device,
    dynamic_weight_module: DynamicWeightingModule,
    output_dim: int,
    circuit_depth: int,
    noise_param: torch.Tensor,
    entanglement_type: str,
    encoding_method: str,
    noise_model: str,
    feature_extractor: Callable = None,
) -> torch.Tensor:
    """
    - Quanvolutional forward pass combining CNN and quantum circuit outputs
    - Uses CNN intermediate features as input to the quantum circuit

    Args:
        input_data: Input tensor to be processed by the CNN
        cnn_model: Classical CNN model
        device: Torch device to use for computation
        dynamic_weight_module: Module for dynamic weighting of classical/quantum outputs
        feature_extractor: Function to extract intermediate features from CNN
        output_dim: Dimension of the output (number of classes)
        circuit_depth: Depth of the entanglement layer of the quantum circuit
        noise_param: Parameter for quantum noise
        entanglement_type: Type of entanglement to use in the quantum circuit
        encoding_method: Method for encoding classical features into quantum states ('angle', 'enhanced_angle')
        noise_model: Type of noise model to use ('depolarizing', 'amplitude_damping', 'phase_damping'. 'mixed')

    Returns:
        Hybrid output combining classical and quantum results
    """

    # Move input data to the specified device
    input_data = input_data.to(device)

    # Run CNN and extract both output and features
    if isinstance(cnn_model, FeatureExtractor):
        classical_output = cnn_model(input_data)
        cnn_features = cnn_model.get_features()
    else:
        print("No feature extractor provided.")

    # Process the CNN features
    # Flatten if features are not already flattened
    if len(cnn_features.shape) > 2:
        feature_size = cnn_features.size(1) * cnn_features.size(2) * cnn_features.size(3)
        cnn_features = cnn_features.view(batch_size, feature_size)

    # Determine the number of qubits needed
    num_qubits = max(2, int(math.ceil(math.log2(output_dim))))

    # Apply selected encoding method
    encoding_methods = ['angle', 'enhanced_angle']
    if encoding_method not in encoding_methods:
        raise ValueError(f"Invalid encoding method: '{encoding_method}'. Valid options are: {', '.join(encoding_methods)}")

    if encoding_method == 'angle':
        # Simple angle encoding
        rx_angles, ry_angles, rz_angles = angle_encoding_basic(cnn_features, num_qubits)
        use_enhanced_encoding = False

    elif encoding_method == 'enhanced_angle':
        # Enhanced angle encoding with multiple rotation types per qubit
        rx_angles, ry_angles, rz_angles = angle_encoding_enhanced(cnn_features, num_qubits)
        use_enhanced_encoding = True

    # Select the appropriate device
    def create_device():
        try:
            return qml.device("default.mixed", wires=num_qubits)
        except Exception as e:
            raise RuntimeError(f"Failed to create the quantum device. Error: {e}.")

    # Initialize device
    try:
        dev = create_device()
        diff_method = "parameter-shift"
    except RuntimeError as e:
        print(f"Device initialization failed. Error: {e}.")

    # Create quantum circuit based on the encoding method
    if use_enhanced_encoding:
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def quantum_circuit(rx_angles, ry_angles, rz_angles, noise_param):
            # Initial rotation layer
            for i in range(num_qubits):
                qml.RX(rx_angles[i], wires=i)
                qml.RY(ry_angles[i], wires=i)
                qml.RZ(rz_angles[i], wires=i)

            # Repeated blocks for circuit depth
            for d in range(circuit_depth):
                # Apply entanglement layer based on selected type
                apply_entanglement_layer(num_qubits, entanglement_type)

                # Apply appropriate noise model
                if noise_model is not None:
                    apply_noise_operations(range(num_qubits), noise_param, noise_model)

            # Return probabilities
            return qml.probs(wires=range(num_qubits))
    else:
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def quantum_circuit(angles, noise_param):
            # Initial rotation layer
            for i in range(num_qubits):
                qml.RY(angles[i], wires=i)

            # Repeated blocks for circuit depth
            for d in range(circuit_depth):
                # Apply entanglement layer based on selected type
                apply_entanglement_layer(num_qubits, entanglement_type)

                # Apply appropriate noise model
                if noise_model is not None:
                    apply_noise_operations(range(num_qubits), noise_param, noise_model)

            # Return probabilities
            return qml.probs(wires=range(num_qubits))

    quantum_outputs = []

    batch_size = classical_output.size(0)

    for i in range(batch_size):
        # Extract sample-specific angles
        if use_enhanced_encoding:
            rx = rx_angles[i].detach().cpu().numpy()
            ry = ry_angles[i].detach().cpu().numpy()
            rz = rz_angles[i].detach().cpu().numpy()
        else:
            ry = ry_angles[i].detach().cpu().numpy()

        try:
            # Run quantum circuit
            if use_enhanced_encoding:
                result = quantum_circuit(rx, ry, rz, noise_param)
            else:
                result = quantum_circuit(ry, noise_param)

            # Ensure NumPy array format
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()

            # Match output dimension
            if len(result) < output_dim:
                result_padded = np.pad(result, (0, output_dim - len(result)))
            else:
                result_padded = result[:output_dim]

            quantum_outputs.append(result_padded)

        except Exception as e:
            print(f"Error in quantum circuit. Error: {e}")

    # Convert final results to PyTorch tensor
    quantum_output_probabilities = torch.tensor(
        np.array(quantum_outputs),
        dtype=torch.float32,
        device=device
    )

    # Ensure quantum output matches the shape of classical output
    if quantum_output_probabilities.shape[1] < classical_output.shape[1]:
        padding = torch.zeros(batch_size,
                             classical_output.shape[1] - quantum_output_probabilities.shape[1],
                             device=device)
        quantum_output_probabilities = torch.cat([quantum_output_probabilities, padding], dim=1)

    quantum_logits = torch.log(quantum_output_probabilities)

    # Small constant for numerical stability
    epsilon = 1e-8  

    # Quantum-classical hybrid output
    if dynamic_weight_module is not None:
        # Dynamic weighting approach - ensure alpha is in [0,1]
        effective_alpha = torch.clamp(dynamic_weight_module(classical_output, quantum_logits), 0.0, 1.0)

        # Convert to probability amplitudes with better numerical stability
        classical_probs = F.softmax(classical_output, dim=1)
        quantum_probs = quantum_output_probabilities / (quantum_output_probabilities.sum(dim=1, keepdim=True) + epsilon)

        classical_probs = torch.clamp(classical_probs, min=epsilon)
        quantum_probs = torch.clamp(quantum_probs, min=epsilon)

        # Square root for amplitude conversion
        classical_amp = torch.sqrt(classical_probs)
        quantum_amp = torch.sqrt(quantum_probs)

        beta_coef = torch.sqrt(torch.clamp(1 - effective_alpha**2, min=epsilon, max=1.0))

        # Combine at amplitude level with proper normalization factor
        combined_amp = effective_alpha * classical_amp + beta_coef * quantum_amp

        # Back to probabilities
        hybrid_probs = combined_amp**2
        hybrid_probs = hybrid_probs / (hybrid_probs.sum(dim=1, keepdim=True) + epsilon)

        hybrid_probs = torch.clamp(hybrid_probs, min=epsilon)
        
    else:
        # Probabilities
        c_conf = F.softmax(classical_output, dim=1)
        q_conf = quantum_output_probabilities.clamp(min=epsilon)
        q_conf = q_conf / (q_conf.sum(dim=1, keepdim=True) + epsilon)   # Normalized

        # Entropy
        c_entropy = -(c_conf * torch.log(c_conf + epsilon)).sum(dim=1, keepdim=True)
        q_entropy = -(q_conf * torch.log(q_conf + epsilon)).sum(dim=1, keepdim=True)

        # Adaptive weights (trust lower entropy more)
        total_entropy = c_entropy + q_entropy + epsilon
        c_weight = q_entropy / total_entropy
        q_weight = c_entropy / total_entropy

        # Amplitude-level combination
        c_amp = torch.sqrt(c_conf)
        q_amp = torch.sqrt(q_conf)

        c_factor = torch.sqrt(c_weight)
        q_factor = torch.sqrt(q_weight)

        # Normalize factors to avoid scaling drift
        norm_factor = torch.sqrt(c_factor**2 + q_factor**2 + epsilon)
        c_factor = c_factor / norm_factor
        q_factor = q_factor / norm_factor

        # Combine amplitudes and back to probabilities
        combined_amp = c_factor * c_amp + q_factor * q_amp
        hybrid_probs = combined_amp**2
        hybrid_probs = hybrid_probs / (hybrid_probs.sum(dim=1, keepdim=True) + epsilon)

    hybrid_output = torch.log(hybrid_probs)

    return hybrid_output

class QNN(nn.Module):
    """
    Quanvolutional Neural Network combining classical CNN with quantum circuits
    """
    def __init__(
        self,
        cnn_model: nn.Module,
        device: torch.device,
        output_dim: int,
        circuit_depth: int,
        noise_strength: float,
        input_noise_injection: bool,
        entanglement_type: str,
        use_dynamic_weights: bool,
        encoding_method: str,
        noise_model: str,
        feature_layer: str = None,
    ):

        """
        Initialize the Quanvolutional Neural Network

        Args:
            cnn_model: The classical CNN model to use
            device: The device to run computations on
            output_dim: Output dimension (number of classes)
            circuit_depth: Depth of the quantum circuit
            noise_strength: Initial strength of quantum noise (0.0 to 1.0)
            input_noise_injection: Enable random perturbations to input data
            entanglement_type: Type of entanglement to use in the quantum circuit
            use_dynamic_weights: Whether to use dynamic weighting between classical and quantum outputs
            encoding_method: Method for encoding classical features into quantum states  ('angle', 'enhanced_angle')
            noise_model: Type of noise model to use in quantum simulation ('depolarizing', 'amplitude_damping', 'phase_damping'. 'mixed')
            feature_layer: Name of the CNN layer to extract features from
        """
        super(QNN, self).__init__()

        # Store model attributes
        self.output_dim = output_dim
        self.device = device
        self.circuit_depth = circuit_depth
        self.input_noise_injection = input_noise_injection
        self.entanglement_type = entanglement_type
        self.use_dynamic_weights = use_dynamic_weights
        self.encoding_method = encoding_method
        self.noise_model = noise_model

        # Create a feature extractor from the CNN model
        self.cnn_model = FeatureExtractor(cnn_model, layer_name=feature_layer)

        # Calculate minimum number of qubits needed
        self.num_qubits = max(2, int(math.ceil(math.log2(output_dim))))

        # Noise strength
        self.raw_noise = torch.tensor(noise_strength)

        # Create enhanced dynamic weighting module if enabled
        if use_dynamic_weights:
            self.dynamic_weight_module = DynamicWeightingModule()
        else:
            self.dynamic_weight_module = None

        # Move the model to the specified device if it's not None
        if device is not None:
            self.to(device)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quanvolutional model

        Args:
            input_data: Input data tensor
            defense_mode: Whether to apply adversarial defense mechanisms

        Returns:
            Hybrid classical-quantum output
        """

        # Apply input randomization (optional)
        if self.input_noise_injection:
            noise_level = 0.01
            input_noise = torch.randn_like(input_data) * noise_level
            input_data = input_data + input_noise

        return hybrid_forward(
            input_data=input_data,
            cnn_model=self.cnn_model,
            device=self.device,
            dynamic_weight_module=self.dynamic_weight_module,
            output_dim=self.output_dim,
            circuit_depth=self.circuit_depth,
            noise_param=self.raw_noise,
            entanglement_type=self.entanglement_type,
            encoding_method=self.encoding_method,
            noise_model=self.noise_model
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the QNN model configuration
        """
        # Add current configuration to metadata
        config = {
            "output_dim": self.output_dim,
            "circuit_depth": self.circuit_depth,
            "num_qubits": self.num_qubits,
            "noise_strength": self.raw_noise.item(),
            "input_noise_injection": self.input_noise_injection,
            "entanglement_type": self.entanglement_type,
            "use_dynamic_weights": self.use_dynamic_weights,
            "encoding_method": self.encoding_method,
            "noise_model": self.noise_model
        }

        return {"config": config}
