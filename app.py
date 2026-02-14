import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
from model import ImageCaptioningModel, Vocabulary

# Page config
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🖼️ Image Caption Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and get AI-generated captions!</p>', unsafe_allow_html=True)

# Load model and vocab
@st.cache_resource
def load_model_and_vocab():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load preprocessed data
    with open('flickr30k_preprocessed.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)
    vocab = preprocessed_data['vocab']
    
    # Load model config
    with open('model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    
    # Load model
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    
    model = ImageCaptioningModel(
        vocab_size=model_config['vocab_size'],
        embed_size=model_config['embed_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=0.0,  # No dropout for inference
        pad_idx=preprocessed_data['pad_idx'],
        start_idx=preprocessed_data['start_idx'],
        end_idx=preprocessed_data['end_idx']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load ResNet50 for feature extraction
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    
    return model, vocab, resnet, device

# Load resources
with st.spinner('🔄 Loading model...'):
    model, vocab, resnet, device = load_model_and_vocab()
    st.success('✅ Model loaded successfully!')

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    generation_method = st.radio(
        "Generation Method",
        options=["Beam Search", "Greedy"],
        help="Beam Search: Better quality, slower\nGreedy: Faster, simpler"
    )
    
    if generation_method == "Beam Search":
        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=3, 
                             help="Higher = better quality but slower")
    else:
        beam_size = 1
    
    max_length = st.slider("Max Caption Length", min_value=10, max_value=100, value=50)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info(f"""
    - **Architecture:** ResNet50 + LSTM
    - **Vocabulary Size:** {len(vocab):,}
    - **Device:** {device}
    - **Training Dataset:** Flickr30k
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    st.subheader("📝 Generated Caption")
    
    if uploaded_file is not None:
        with st.spinner('🤖 Generating caption...'):
            try:
                # Preprocess image
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    features = resnet(image_tensor).squeeze()
                
                # Generate caption
                method = 'beam' if generation_method == "Beam Search" else 'greedy'
                caption = model.generate_caption(
                    features.unsqueeze(0),
                    vocab,
                    max_length=max_length,
                    method=method,
                    beam_size=beam_size
                )
                
                # Display caption
                st.markdown(f"""
                    <div style='
                        background-color: #f0f8ff;
                        padding: 2rem;
                        border-radius: 10px;
                        border-left: 5px solid #1f77b4;
                        margin-top: 1rem;
                    '>
                        <h3 style='color: #1f77b4; margin-bottom: 1rem;'>Caption:</h3>
                        <p style='font-size: 1.2rem; color: #333;'>{caption}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Caption generated successfully!")
                
                # Download button
                st.download_button(
                    label="💾 Download Caption",
                    data=caption,
                    file_name="caption.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload an image to generate a caption")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | Powered by PyTorch | Trained on Flickr30k Dataset</p>
    </div>
""", unsafe_allow_html=True)