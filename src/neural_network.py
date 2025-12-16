import numpy as np
from typing import Optional, List, Tuple, Union

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    # Derivative is 1 for x > 0, 0 for x <= 0 (sub-gradient convention)
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),           # x >= 0: standard formula
        np.exp(x) / (1 + np.exp(x))     # x < 0: equivalent but stable
    )


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig * (1 - sig)


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


class DenseLayer:

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'relu',
        random_seed: Optional[int] = None
    ):
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation.lower()
        
        # Validate activation function
        valid_activations = ['relu', 'sigmoid', 'linear']
        if self.activation not in valid_activations:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from: {valid_activations}"
            )
        
        # Determine initialization standard deviation based on activation
        if self.activation == 'relu':
            # Kaiming initialization for ReLU (std = sqrt(2/input_size))
            std = np.sqrt(2.0 / input_size)
        elif self.activation in ('sigmoid', 'linear'):
            # Simplified Xavier/Glorot for Sigmoid/Linear (std = sqrt(1/input_size))
            std = np.sqrt(1.0 / input_size)
        else:
            # Fallback (should not be reached due to validation)
            std = 0.01 
            
        self.weights = np.random.randn(output_size, input_size) * std
        
        # Initialize biases to zero
        self.biases = np.zeros(output_size)
        

        # CACHE VARIABLES - Stored during forward pass, used in backward pass
        
        self.last_input = None          # Input x from previous layer
        self.last_z = None              # Linear output (W @ x + b)
        self.last_activation = None     # Activation output
        
        
        # GRADIENT VARIABLES - Computed during backward pass
        
        self.d_weights = None           # Gradient of loss w.r.t. weights
        self.d_biases = None            # Gradient of loss w.r.t. biases
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Store input for use in backward pass
        self.last_input = x
        
        # Linear calculation: X @ W.T + b
        self.last_z = x @ self.weights.T + self.biases  # Shape: (batch_size, output_size)
        
        # Apply activation
        if self.activation == 'relu':
            self.last_activation = relu(self.last_z)
        elif self.activation == 'sigmoid':
            self.last_activation = sigmoid(self.last_z)
        elif self.activation == 'linear':
            self.last_activation = linear(self.last_z)
        
        return self.last_activation
    
    def backward(self, d_out: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        
        batch_size = d_out.shape[0]
        
        
        # Compute dL/dz (gradient w.r.t. pre-activation)
        # dL/dz = dL/da * da/dz
        
        if self.activation == 'relu':
            d_z = d_out * relu_derivative(self.last_z)
        elif self.activation == 'sigmoid':
            d_z = d_out * sigmoid_derivative(self.last_z)
        elif self.activation == 'linear':
            d_z = d_out * linear_derivative(self.last_z)
        

        # Compute dL/dW (gradient w.r.t. weights) - Shape (output_size, input_size)
        # dL/dW = (1/N) * dL/dZ.T @ X
        self.d_weights = (d_z.T @ self.last_input) / batch_size
        
        
        # Compute dL/db (gradient w.r.t. biases) - Shape (output_size,)
        # dL/db = (1/N) * sum(dL/dZ, axis=0)
        self.d_biases = np.sum(d_z, axis=0) / batch_size
        
        
        # Compute dL/dx (gradient w.r.t. input) - Shape (batch_size, input_size)
        # dL/dX = dL/dZ @ W (W has shape (output_size, input_size))
        d_input = d_z @ self.weights
        
        
        # Update weights and biases (gradient descent)
        
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
        
        # Return gradient w.r.t. input (for previous layer)
        return d_input
    
    def __repr__(self) -> str:
        return (f"DenseLayer(input_size={self.input_size}, "
                f"output_size={self.output_size}, activation='{self.activation}')")


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    # Ensure both arrays have the same shape
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Compute squared differences and return mean
    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)


def mse_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    
    # Ensure both arrays have the same shape
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Return per-sample gradient (averaging is done in layer backward pass)
    # dL/dy = 2 * (y_pred - y_true) for each sample
    gradient = 2.0 * (y_pred - y_true)
    
    return gradient

class NeuralNetwork:
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        hidden_activation: str = 'relu',
        random_seed: Optional[int] = None
    ):
        
        
        # Validate inputs
        
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes must have at least 2 elements (input and output), "
                f"got {len(layer_sizes)}"
            )
        
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(
                f"All layer sizes must be positive integers, got {layer_sizes}"
            )
        
        # Store hyperparameters
    
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation.lower()
        self.random_seed = random_seed
        
        # Validate activation
        if self.hidden_activation not in ['relu', 'sigmoid']:
            raise ValueError(
                f"hidden_activation must be 'relu' or 'sigmoid', "
                f"got '{hidden_activation}'"
            )
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        
        # Create layers
        
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Determine activation for this layer
            if i == len(layer_sizes) - 2:
                # Last layer (output): always linear for regression (MSE)
                activation = 'linear'
            else:
                # Hidden layers: use specified activation
                activation = self.hidden_activation
            
            # Create the layer
            layer = DenseLayer(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
            )
            
            self.layers.append(layer)
        
        # Initialize training history
        
        self.loss_history = []  # Will store average loss for each epoch
        
        print("NEURAL NETWORK INITIALIZED")
        print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
        print(f"  Input layer:  {layer_sizes[0]} features")
        for i, size in enumerate(layer_sizes[1:-1], 1):
            print(f"  Hidden layer {i}: {size} neurons ({self.hidden_activation})")
        print(f"  Output layer: {layer_sizes[-1]} neuron(s) (linear)")
        print(f"\nHyperparameters:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs:        {self.epochs}")
        print(f"  Batch size:    {self.batch_size}")
        print(f"  Random seed:   {self.random_seed}")
        
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
                
        current_input = X
        
        # Pass through each layer sequentially
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        # current_input is now the final output (y_pred)
        return current_input
    
    def _backward_pass(self, loss_gradient: np.ndarray) -> None:
        
        current_gradient = loss_gradient
        
        # Propagate backwards through layers (reverse order)
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, self.learning_rate)
        
    
    def fit(
        self,
        X: Union[np.ndarray, 'pd.DataFrame'],
        y: Union[np.ndarray, 'pd.Series'],
        verbose: int = 1
    ) -> 'NeuralNetwork':
                
        # Convert inputs to numpy arrays and validate
        
        # Handle pandas DataFrames/Series (for compatibility, though only NumPy is used)
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Ensure numpy arrays and use float64 for precision
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Ensure y is 2D (n_samples, 1) for consistent matrix math
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Validate shapes
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        # Validate that input size matches network architecture
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but network expects "
                f"{self.layer_sizes[0]} features (first layer size)"
            )
        
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        
        # Training loop
        
        print("\nTRAINING STARTED")
        print(f"Training samples: {n_samples:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batches per epoch: {n_batches:,}")
        print(f"Total epochs: {self.epochs}")
        
        for epoch in range(self.epochs):
            
            # Shuffle data at the start of each epoch
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            
            # Process each mini-batch
            
            for batch_idx in range(n_batches):
                # Get batch data
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                
                # FORWARD PASS: Compute predictions
                
                y_pred = self._forward_pass(X_batch)
                
                
                # COMPUTE LOSS
                
                batch_loss = mse_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                
                # COMPUTE LOSS GRADIENT
                
                loss_grad = mse_loss_derivative(y_batch, y_pred)
                
                
                # BACKWARD PASS: Update weights
                
                self._backward_pass(loss_grad)
            
            
            # Record average loss for this epoch
            
            avg_epoch_loss = epoch_loss / n_batches
            self.loss_history.append(avg_epoch_loss)
            
            
            # Print progress
            
            
            if verbose == 2:
                # Print every epoch
                print(f"Epoch {epoch+1:3d}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
            elif verbose == 1 and (epoch + 1) % 10 == 0:
                # Print every 10 epochs
                print(f"Epoch {epoch+1:3d}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
        
        
        # Training complete
        
        print("\nTRAINING COMPLETED")
        
        if self.loss_history:
            print(f"Final loss: {self.loss_history[-1]:.6f}")
            print(f"Initial loss: {self.loss_history[0]:.6f}")
            print(f"Improvement: {self.loss_history[0] - self.loss_history[-1]:.6f}")
        
        
        
        return self  # Return self for method chaining
    
    def predict(self, X: Union[np.ndarray, 'pd.DataFrame']) -> np.ndarray:
        
        
        # Handle pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Validate shape
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but network expects "
                f"{self.layer_sizes[0]} features"
            )
        
        # Forward pass only (no training)
        predictions = self._forward_pass(X)
        
        # Flatten if output size is 1 (common for regression)
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def __repr__(self) -> str:
        return (f"NeuralNetwork(architecture={' -> '.join(map(str, self.layer_sizes))}, "
                f"learning_rate={self.learning_rate}, epochs={self.epochs})")