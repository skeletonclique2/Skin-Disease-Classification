import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms
import torch.nn.functional as F
import time
from torch import nn

# Load the models
@st.cache_resource
def load_model(model_path, model_type):
    if model_type == "Model 1":
        model = timm.create_model('efficientnet_b0', pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 7)
    elif model_type == "Model 2":
        class EnhancedHybridModel(nn.Module):
            def __init__(self, num_classes):
                super(EnhancedHybridModel, self).__init__()
        
                # EfficientNet Backbone
                self.efficientnet = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
                efficientnet_out_features = self.efficientnet.num_features
        
                # ResNet Backbone
                self.resnet = timm.create_model('resnet50', pretrained=False, num_classes=0)
                resnet_out_features = self.resnet.num_features
        
                # Attention Layer
                self.attention = nn.Sequential(
                    nn.Linear(efficientnet_out_features + resnet_out_features, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
        
                # Fully connected classification layer
                self.classifier = nn.Linear(efficientnet_out_features + resnet_out_features, num_classes)

            def forward(self, x):
                # Extract features from EfficientNet
                efficientnet_features = self.efficientnet(x)
        
                # Extract features from ResNet
                resnet_features = self.resnet(x)
        
                # Concatenate features
                combined_features = torch.cat((efficientnet_features, resnet_features), dim=1)
        
                # Apply attention
                attention_weights = self.attention(combined_features)
                combined_features = combined_features * attention_weights
        
                # Pass through the classifier
                output = self.classifier(combined_features)
                return output

            
        
        model = EnhancedHybridModel(num_classes=7)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing
def preprocess_image(image_path):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Prediction
def predict(model, input_tensor, class_names):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        return class_names[predicted_class.item()], confidence.item()

# Streamlit UI
st.title("Skin Disease Classifier")
st.write("Welcome to our Skin Disease Classifier DL Model")
st.write("**by HARIS AAMIR & MUHAMMAD HANZALA PAREKH**")

# Simulate loading animation
with st.spinner("Loading..."):
    time.sleep(2)  # Simulate loading time

# Transition to the next screen
st.header("Select a Model")
model_choice = st.radio("Choose the model you want to use for prediction:", ["Model 1", "Model 2"])

if model_choice:
    # Load the appropriate model
    model_path = "/Users/a/Library/CloudStorage/OneDrive-HigherEducationCommission/Semester 7/Deep Learning/Ass 2/fine_tuned_model.pth" if model_choice == "Model 1" else "/Users/a/Library/CloudStorage/OneDrive-HigherEducationCommission/Semester 7/Deep Learning/Ass 3/hybrid_model_best.pth"
    model = load_model(model_path, model_type=model_choice)
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # Upload an image
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        input_tensor = preprocess_image(uploaded_file)
        predicted_class, confidence = predict(model, input_tensor, class_names)

        # Display results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.write(f"**Model Used:** {'EfficientNet' if model_choice == 'Model 1' else 'EfficientNet + ResNet'}")
