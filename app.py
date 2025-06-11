import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# DÒNG QUAN TRỌNG 1: Import lớp Attention chính thức từ Keras
from tensorflow.keras.layers import Attention 
import tensorflow as tf


# --- CÁC HÀM XỬ LÝ ---

@st.cache_resource
def load_all_models():
    # DÒNG QUAN TRỌNG 2: Khai báo để Keras biết "Attention" là gì khi tải model
    custom_objects = {'Attention': Attention}
    
    try:
        # Tải mô hình với custom_objects
        model = load_model('best_model_with_attention.h5', custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Lỗi khi tải file model 'best_model_with_attention.h5': {e}")
        st.info("Hãy chắc chắn rằng bạn đã định nghĩa đúng lớp Attention hoặc mô hình không bị lỗi.")
        return None, None, None

    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Không tìm thấy file 'tokenizer.pkl'. Vui lòng tải file lên.")
        return None, None, None
        
    feature_model = ResNet50(weights='imagenet')
    feature_model = tf.keras.Model(inputs=feature_model.inputs, outputs=feature_model.layers[-2].output)
    
    return model, tokenizer, feature_model

def extract_feature(image, feature_model):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_feature, tokenizer, max_length=34):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_feature, sequence], verbose=0)
        pred_id = np.argmax(yhat)
        word = idx_to_word(pred_id, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
            
    final_caption = ' '.join(in_text.split()[1:])
    return final_caption.capitalize() + '.'

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Chuyển đổi Hình ảnh thành Văn bản", layout="centered")

st.title("️Chuyển đổi Hình ảnh thành Văn bản 🖼️➡️📝")
st.write("Tải lên một hình ảnh và mô hình sẽ tạo ra một chú thích mô tả nội dung của ảnh.")

model, tokenizer, feature_model = load_all_models()

uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn đã tải lên', use_column_width=True)
    st.write("")

    if st.button('Tạo Chú Thích', use_container_width=True):
        with st.spinner('Mô hình đang phân tích ảnh, vui lòng chờ...'):
            image_feature = extract_feature(image, feature_model)
            caption = predict_caption(model, image_feature, tokenizer)
            st.subheader("Chú thích được tạo ra:")
            st.success(caption)
else:
    if model is None:
        st.warning("Không thể tải được mô hình. Vui lòng kiểm tra lại file.")
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")
