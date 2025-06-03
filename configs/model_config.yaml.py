# configs/model_config.yaml
model_name: "SimpleCNN"
architecture:
num_classes: 2  # Binary classification (e.g., pathology vs no pathology)
input_channels: 1  # Grayscale medical images
# Specific layers for SimpleCNN can be defined here or hardcoded in the model
conv_layers:
- out_channels: 16
kernel_size: 3
stride: 1
padding: 1
- out_channels: 32
kernel_size: 3
stride: 1
padding: 1
fc_layers:
- out_features: 128

training_params:
learning_rate: 0.001
batch_size: 8  # Small batch for initial training
num_epochs_initial: 10
num_epochs_active_
retraining: 5
optimizer: "Adam"
loss_function: "CrossEntropyLoss"

device: "cuda"  # "cuda" if available, else "cpu"

