# File: app.py (version 5 - Buộc dùng CPU để gỡ lỗi)
import os
# !!! THAY ĐỔI QUAN TRỌNG !!!
# Đặt dòng này LÊN TRÊN CÙNG, trước tất cả các lệnh import của tensorflow/keras
# Dòng này sẽ vô hiệu hóa GPU và buộc TensorFlow chạy trên CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
from PIL import Image
import numpy as np
import pickle
from gtts import gTTS
import io

# Import các thành phần từ Keras 3
import tensorflow as tf
import keras
from keras.models import load_model, Model
from keras.utils import img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Attention 


# --- CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="Tạo Chú thích Ảnh")

# --- THAM SỐ VÀ CÀI ĐẶT ---
MODEL_PATH = 'best_model_with_attention.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_CAPTION_LENGTH = 41

@st.cache_resource
def load_keras_model_and_tokenizer():
    """Tải model Keras và tokenizer, đồng thời đăng ký lớp Attention có sẵn."""
    try:
        st.info("Bắt đầu tải tokenizer...")
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        st.info("Tải tokenizer thành công.")

        custom_objects = {'Attention': Attention}
        
        st.info("Bắt đầu tải model chính (best_model_with_attention.h5)...")
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        st.info("Tải model chính thành công.")
        
        st.info("Bắt đầu tải model ResNet50 (trích xuất đặc trưng)...")
        base_model = ResNet50(weights='imagenet')
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        st.info("Tải ResNet50 thành công.")

        st.success("Tải tất cả model và tokenizer hoàn tất!")
        return model, tokenizer, feature_extractor
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tải model hoặc tokenizer: {e}")
        st.stop()

# --- CÁC HÀM XỬ LÝ ---
def preprocess_image_for_resnet(image_pil):
    img = image_pil.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def generate_caption_keras(image_pil, model, tokenizer, feature_extractor):
    in_text = 'startseq'
    preprocessed_img = preprocess_image_for_resnet(image_pil)
    image_features = feature_extractor.predict(preprocessed_img)

    for _ in range(MAX_CAPTION_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LENGTH)[0]
        sequence_input = np.array([sequence])
        prediction = model.predict([image_features, sequence_input], verbose=0)
        predicted_word_idx = np.argmax(prediction[0])
        word = tokenizer.index_word.get(predicted_word_idx, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    
    final_caption = in_text.replace('startseq ', '').replace(' endseq', '').replace('_', ' ')
    return final_caption.capitalize()

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='vi', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Lỗi khi tạo audio: {e}")
        return None

# --- GIAO DIỆN ---
caption_model, tokenizer, feature_extractor = load_keras_model_and_tokenizer()
st.title("️🖼️ Trình tạo Chú thích ảnh Tự động")
st.markdown("---")
col1, col2 = st.columns(spec=[1, 1], gap="large")
with col1:
    st.header("1. Tải ảnh của bạn lên")
    uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    image_placeholder = st.empty()
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_placeholder.image(image, caption="Ảnh đã tải lên", use_container_width=True)
with col2:
    st.header("2. Kết quả chú thích")
    if uploaded_file:
        with st.spinner("🧠 Mô hình đang suy nghĩ..."):
            caption_result = generate_caption_keras(image, caption_model, tokenizer, feature_extractor)
        st.info("Chú thích được tạo ra là:")
        st.markdown(f"### *✨ \"{caption_result}\"*") 
        st.markdown("---")
        st.subheader("Nghe chú thích")
        with st.spinner("🔊 Đang tạo âm thanh..."):
            audio_data = text_to_speech(caption_result)
        if audio_data:
            st.audio(audio_data, format='audio/mp3')
    else:
        st.warning("Vui lòng tải ảnh lên ở cột bên trái để xem kết quả.")
