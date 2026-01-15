import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import requests
import io

# ────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="ResNet-18 Real-time Classifier",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Real-time Image Classification with ResNet-18")
st.markdown("""
This app uses a **pre-trained ResNet-18** model to classify images from your webcam or uploaded files  
into one of 1,000 ImageNet classes.  
Choose **Take Photo** or **Upload Image** below.
""")

# ────────────────────────────────────────────────
# Load model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

model = load_model()

# ────────────────────────────────────────────────
# Image preprocessing
# ────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ────────────────────────────────────────────────
# Load ImageNet class names (cached)
# ────────────────────────────────────────────────
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        labels = [line.strip() for line in response.text.splitlines() if line.strip()]
        return labels
    except Exception as e:
        st.error(f"Could not download labels: {e}")
        return ["class_" + str(i) for i in range(1000)]  # fallback

labels = load_imagenet_labels()

# ────────────────────────────────────────────────
# Input selection
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Take Photo (Webcam)", "📁 Upload Image"])

image = None

with tab1:
    st.info("Click **Take Photo** to capture from webcam")
    camera_input = st.camera_input(" ", key="camera", label_visibility="collapsed")
    if camera_input is not None:
        image = Image.open(io.BytesIO(camera_input.getvalue())).convert("RGB")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload an image (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"],
        help="Drag & drop or click to browse"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# ────────────────────────────────────────────────
# Inference when we have an image
# ────────────────────────────────────────────────
if image is not None:
    # Show the image
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    # Softmax + top 5
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    top5_prob = top5_prob.cpu().numpy()
    top5_idx = top5_idx.cpu().numpy()
    top5_labels = [labels[i] for i in top5_idx]

    # ── Results ─────────────────────────────────────
    st.subheader("🔍 Top 5 Predictions")
    
    # List view
    for i in range(5):
        prob_pct = top5_prob[i] * 100
        st.write(f"**{i+1}. {top5_labels[i]}**  —  {prob_pct:.2f}%  ({top5_prob[i]:.4f})")

    # Bar chart
    st.subheader("📊 Confidence Chart")
    chart_df = pd.DataFrame({
        "Class": top5_labels,
        "Confidence": top5_prob * 100
    })
    st.bar_chart(chart_df.set_index("Class"), y="Confidence", height=300)

else:
    st.info("👆 Take a photo with your webcam or upload an image to start classification.")

# Footer note
st.markdown("---")
st.caption("Powered by PyTorch • ResNet-18 (ImageNet) • Streamlit • No GPU required")