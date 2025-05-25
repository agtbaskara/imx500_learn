import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Updated transform to resize MNIST to 224x224 RGB images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels (RGB)
    transforms.Lambda(lambda x: 1 - x)  # Invert colors
])

# Download and load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)  # Reduced batch size due to larger images

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)  # Reduced batch size

# Define a CNN for 224x224 RGB images
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: 3x224x224
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Output: 16x112x112
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x56x56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32x28x28
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x14x14
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x14x14
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the neural network, loss function, and optimizer
model = CNN().to(device)  # Move the model to the selected device (GPU or CPU)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed to Adam for better convergence

# Training loop
epochs = 1
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Move inputs and labels to the selected device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        
        # Print statistics every 100 mini-batches
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch+1} completed")

    # Validate after each epoch
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation accuracy after epoch {epoch+1}: {100 * correct / total:.2f}%')

# Final evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final accuracy on the test set: {100 * correct / total:.2f}%')

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Updated dummy input size
torch.onnx.export(model,                      # model being run
                  dummy_input,                # model input
                  "mnist_224x224_rgb.onnx",   # updated filename
                  input_names=["input"],      # input names
                  output_names=["output"],    # output names
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Model exported to mnist_224x224_rgb.onnx")

# Quantization using Sony Model Compression Toolkit (MCT)
try:
    import model_compression_toolkit as mct

    # Define a representative dataset generator for MCT
    def representative_dataset_gen():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= 10:  # Limit to 10 iterations
                break
            yield [inputs.to(device)]

    # Define the target platform capabilities for quantization
    target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

    # Apply post-training quantization
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap
    )

    # Evaluate quantized model
    print("\nEvaluating quantized model:")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = quantized_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Quantized model accuracy: {accuracy:.2f}%')

    # Export the quantized model to ONNX
    mct.exporter.pytorch_export_model(
        quantized_model, 
        save_model_path='quantized_mnist_224x224_rgb.onnx', 
        repr_dataset=representative_dataset_gen
    )
    print("Quantized model exported to quantized_mnist_224x224_rgb.onnx")
    
except ImportError:
    print("Sony Model Compression Toolkit (MCT) not found. Skipping quantization step.")