import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# --- THAY ĐỔI QUAN TRỌNG ---
# 1. Xóa bỏ hoàn toàn phần "class Attention(Layer): ..." mà chúng ta tự viết.
# 2. Import trực tiếp lớp Attention chính thức từ thư viện Keras.
from tensorflow.keras.layers import Attention
import tensorflow as tf


# --- CÁC HÀM XỬ LÝ ---

# Hàm tải các model và tokenizer (được cache lại để chạy nhanh hơn)
@st.cache_resource
def load_all_models():
    # Vẫn cần khai báo custom_objects để Keras biết ánh xạ tên "Attention" 
    # trong file model tới lớp Attention chính thức vừa import.
    custom_objects = {'Attention': Attention}
    
    # Tải mô hình chính
    try:
        model = load_model('best_model_with_attention.h5', custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Lỗi khi tải file model 'best_model_with_attention.h5': {e}")
        st.info("Hãy chắc chắn rằng bạn đã định nghĩa đúng lớp Attention hoặc mô hình không bị lỗi.")
        return None, None, None

    # Tải tokenizer
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Không tìm thấy file 'tokenizer.pkl'. Vui lòng tải file lên.")
        return None, None, None
        
    # Tải mô hình ResNet50 để trích xuất đặc trưng ảnh
    feature_model = ResNet50(weights='imagenet')
    # Bỏ lớp cuối cùng để lấy vector đặc trưng
    feature_model = tf.keras.Model(inputs=feature_model.inputs, outputs=feature_model.layers[-2].output)
    
    return model, tokenizer, feature_model

def extract_feature(image, feature_model):
    image = image.resize((224, 224))
    image = np.array(image)
    # Nếu ảnh có 4 kênh (PNG), bỏ kênh alpha
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
        
        # Dự đoán
        yhat = model.predict([image_feature, sequence], verbose=0)
        
        # Lấy từ có xác suất cao nhất
        pred_id = np.argmax(yhat)
        word = idx_to_word(pred_id, tokenizer)
        
        if word is None:
            break
        
        in_text += ' ' + word
        
        if word == 'endseq':
            break
            
    # Dọn dẹp câu kết quả
    final_caption = in_text.split()
    final_caption = final_caption[1:-1] # Bỏ 'startseq' và 'endseq'
    final_caption = ' '.join(final_caption)
    return final_caption.capitalize() + '.'

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Chuyển đổi Hình ảnh thành Văn bản", layout="centered")

st.title("️Chuyển đổi Hình ảnh thành Văn bản 🖼️➡️📝")
st.write("Tải lên một hình ảnh và mô hình sẽ tạo ra một chú thích mô tả nội dung của ảnh.")

# Tải model
model, tokenizer, feature_model = load_all_models()

uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn đã tải lên', use_column_width=True)
    st.write("")

    if st.button('Tạo Chú Thích', use_container_width=True):
        with st.spinner('Mô hình đang phân tích ảnh, vui lòng chờ...'):
            # Trích xuất đặc trưng ảnh
            image_feature = extract_feature(image, feature_model)
            
            # Tạo chú thích
            caption = predict_caption(model, image_feature, tokenizer)
            
            st.subheader("Chú thích được tạo ra:")
            st.success(caption)
else:
    if model is None:
        st.warning("Không thể tải được mô hình. Vui lòng kiểm tra lại file.")
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")
