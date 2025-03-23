import streamlit as st
import PyPDF2
import pandas as pd
import torch
import os
import re
import base64  # For CSV download

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import transformers: {str(e)}. Please install it with `pip install transformers`.")
    TRANSFORMERS_AVAILABLE = False

# Set page configuration
st.set_page_config(page_title="WizNerd Insp", page_icon="🚀", layout="centered")

# Custom CSS for Tw Cen MT font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tw+Cen+MT&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    .stTable table {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Model name
MODEL_NAME = "amiguel/instruct_BERT-base-uncased_model"

# Label mapping
LABEL_TO_CLASS = {
    0: "Campaign", 1: "Corrosion Monitoring", 2: "Flare Tip", 3: "Flare TIP",
    4: "FU Items", 5: "Intelligent Pigging", 6: "Lifting", 7: "Non Structural Tank",
    8: "Piping", 9: "Pressure Safety Device", 10: "Pressure Vessel (VIE)",
    11: "Pressure Vessel (VII)", 12: "Structure", 13: "Flame Arrestor"
}

# Title
st.title("🚀 WizNerd Insp 🚀")

# Avatars
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# Sidebar
with st.sidebar:
    st.header("Upload Documents 📂")
    uploaded_file = st.file_uploader(
        "Choose a PDF, XLSX, or CSV file",
        type=["pdf", "xlsx", "csv"],
        label_visibility="collapsed"
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "file_data" not in st.session_state:
    st.session_state.file_data = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# File processing function with cache
@st.cache_data
def process_file(uploaded_file, _cache_key):
    if uploaded_file is None:
        return None
    
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            text = re.sub(r'\s+', ' ', text.lower().strip())
            return {"type": "text", "content": text}
        
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
            df = pd.read_excel(uploaded_file) if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" else pd.read_csv(uploaded_file)
            required_cols = ["Scope", "Functional Location", "Unit name"]  # Unit name now required
            
            # Check if all required columns are present
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}. Please upload a file with 'Scope', 'Functional Location', and 'Unit name'.")
                return None
            
            # Pre-process and concatenate Scope, Functional Location, and Unit name
            df = df.dropna(subset=required_cols)
            df["input_text"] = df[required_cols].apply(
                lambda row: " ".join([re.sub(r'\s+', ' ', str(val).lower().strip()) for val in row]), axis=1
            )
            return {"type": "table", "content": df[["input_text"] + required_cols]}
        
    except Exception as e:
        st.error(f"📄 Error processing file: {str(e)}")
        return None

# Model loading function
@st.cache_resource
def load_model(hf_token):
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        if not hf_token:
            st.error("🔐 Please set the HF_TOKEN environment variable.")
            return None
        login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_TO_CLASS), token=hf_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, tokenizer
    except Exception as e:
        st.error(f"🤖 Model loading failed: {str(e)}")
        return None

# Classification function
def classify_instruction(prompt, context, model, tokenizer):
    model.eval()
    device = model.device
    
    if isinstance(context, pd.DataFrame):
        predictions = []
        for text in context["input_text"]:
            full_prompt = f"Context:\n{text}\n\nInstruction: {prompt}"
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.argmax().item()
                predictions.append(LABEL_TO_CLASS[prediction])
        return predictions
    else:
        full_prompt = f"Context:\n{context}\n\nInstruction: {prompt}"
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()
        return LABEL_TO_CLASS[prediction]

# CSV download function
def get_csv_download_link(df, filename="predicted_classes.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Load model
if "model" not in st.session_state:
    model_data = load_model(HF_TOKEN)
    if model_data is None and TRANSFORMERS_AVAILABLE:
        st.error("Failed to load model. Check HF_TOKEN.")
        st.stop()
    elif TRANSFORMERS_AVAILABLE:
        st.session_state.model, st.session_state.tokenizer = model_data

model = st.session_state.get("model")
tokenizer = st.session_state.get("tokenizer")

# Check for new file upload and clear cache
if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.cache_data.clear()  # Clear all cached data
    st.session_state.file_processed = False
    st.session_state.file_data = None
    st.session_state.last_uploaded_file = uploaded_file

# Process uploaded file once
if uploaded_file and not st.session_state.file_processed:
    cache_key = f"{uploaded_file.name}_{uploaded_file.size}"
    file_data = process_file(uploaded_file, cache_key)
    if file_data:
        st.session_state.file_data = file_data
        st.session_state.file_processed = True
        if file_data["type"] == "table":
            st.write("File uploaded with Scope, Functional Location, and Unit name data. Please provide an instruction.")
        else:
            st.write("File uploaded as text context. Please provide an instruction.")

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input handling
if prompt := st.chat_input("Ask your inspection question..."):
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not available.")
        st.stop()

    # Add user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Handle response
    if model and tokenizer:
        try:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                if st.session_state.file_data:
                    file_data = st.session_state.file_data
                    if file_data["type"] == "table":
                        predictions = classify_instruction(prompt, file_data["content"], model, tokenizer)
                        result_df = file_data["content"][["Scope", "Functional Location", "Unit name"]].copy()
                        result_df["Predicted Class"] = predictions
                        st.write("Predicted Item Classes:")
                        st.table(result_df)
                        st.markdown(get_csv_download_link(result_df), unsafe_allow_html=True)
                        response = "Predictions completed for uploaded file."
                    else:
                        predicted_class = classify_instruction(prompt, file_data["content"], model, tokenizer)
                        response = f"The Item Class is: {predicted_class}"
                else:
                    # Handle single prompt without file
                    predicted_class = classify_instruction(prompt, "", model, tokenizer)
                    response = f"The Item Class is: {predicted_class}"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"⚡ Classification error: {str(e)}")
    else:
        st.error("🤖 Model not loaded!")
