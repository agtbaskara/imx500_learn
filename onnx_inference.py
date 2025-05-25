import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
sess = ort.InferenceSession('mnist_224x224_rgb.onnx')

# Helper function to convert logits to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Preprocess the image (load as RGB and normalize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Load image in BGR (OpenCV default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (since the model expects RGB input)
    image = cv2.resize(image, (224, 224))  # Resize to 224x224 pixels

    # Normalize pixel values to range 0-1
    image = image.astype(np.float32) / 255.0
    
    # Ensure the image has the correct shape (1, 3, 224, 224)
    image = image.transpose(2, 0, 1)  # Change shape from (224, 224, 3) to (3, 224, 224)
    return image[np.newaxis, :, :, :].astype(np.float32)  # Add batch dimension (1, 3, 224, 224)

# Run inference
def infer_image(image_path):
    image = preprocess_image(image_path)
    output = sess.run(None, {sess.get_inputs()[0].name: image})

    print(output)
    
    probs = softmax(output[0][0])
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]
    return predicted_class, confidence

# Inference and result
image_path = 'input_tensor_8.png'  # Replace with your image path
predicted_class, confidence = infer_image(image_path)
print(f'Predicted Class: {predicted_class}')
print(f'Confidence: {confidence:.4f}')