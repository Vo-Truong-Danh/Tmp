import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import pickle
from PIL import Image
import os

# --- Thiết lập trang ---
st.set_page_config(page_title="Trình tạo Chú thích Ảnh", layout="centered")

# --- Tải Model và Tokenizer (Sử dụng cache để không tải lại mỗi lần) ---
@st.cache_resource
def load_all_models():
    """
    Tải tất cả các model cần thiết: model tạo caption, tokenizer, và model trích xuất đặc trưng.
    Hàm này được cache để tăng tốc độ.
    """
    # Tải model tạo caption đã được huấn luyện
    # Đảm bảo tệp 'best_model_with_attention.h5' nằm cùng thư mục
    caption_model = load_model('best_model_with_attention.h5')

    # Tải tokenizer
    # Đảm bảo tệp 'tokenizer.pkl' nằm cùng thư mục
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Tải model ResNet50 để trích xuất đặc trưng ảnh
    # Chúng ta chỉ lấy các lớp đến trước lớp phân loại cuối cùng
    image_model = ResNet50(weights='imagenet')
    # Tạo một model mới bằng cách lấy đầu ra của lớp áp chót (thường là GlobalAveragePooling2D)
    feature_extractor = tf.keras.Model(inputs=image_model.inputs, outputs=image_model.layers[-2].output)
    
    return caption_model, tokenizer, feature_extractor

# --- Hàm trích xuất đặc trưng từ ảnh ---
def extract_features(image, model):
    """
    Chuyển đổi ảnh người dùng tải lên thành vector đặc trưng bằng ResNet50.
    """
    # Thay đổi kích thước ảnh thành 224x224 cho phù hợp với ResNet50
    image = image.resize((224, 224))
    # Chuyển ảnh thành mảng numpy
    image_array = img_to_array(image)
    # Thêm một chiều để tạo thành batch size là 1
    image_array = np.expand_dims(image_array, axis=0)
    # Tiền xử lý ảnh theo cách mà ResNet50 yêu cầu
    image_array = preprocess_input(image_array)
    # Trích xuất đặc trưng
    features = model.predict(image_array, verbose=0)
    return features

# --- Hàm chuyển đổi index thành từ ---
def word_for_id(integer, tokenizer):
    """
    Tìm từ tương ứng với một chỉ số (integer) trong tokenizer.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# --- Hàm tạo Caption (Sử dụng thuật toán Greedy Search) ---
def generate_caption(model, tokenizer, photo_features, max_length):
    """
    Tạo ra một câu chú thích từ đặc trưng ảnh.
    """
    # Bắt đầu chuỗi với token 'startseq'
    in_text = 'startseq'
    # Lặp lại để tạo từng từ trong câu
    for _ in range(max_length):
        # Chuyển chuỗi hiện tại thành dạng số
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Padding để chuỗi có độ dài cố định
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Dự đoán từ tiếp theo
        yhat = model.predict([photo_features, sequence], verbose=0)
        # Lấy ra từ có xác suất cao nhất
        yhat = np.argmax(yhat)
        # Chuyển từ dạng số về dạng chữ
        word = word_for_id(yhat, tokenizer)
        # Dừng lại nếu không tìm thấy từ
        if word is None:
            break
        # Thêm từ vừa dự đoán vào chuỗi
        in_text += ' ' + word
        # Dừng lại nếu gặp token 'endseq'
        if word == 'endseq':
            break
            
    # Dọn dẹp câu kết quả
    final_caption = in_text.split()
    final_caption = final_caption[1:-1] # Bỏ 'startseq' và 'endseq'
    final_caption = ' '.join(final_caption)
    return final_caption.capitalize() + '.'

# --- Giao diện người dùng Streamlit ---

st.title("📷 Trình tạo Chú thích Ảnh bằng AI")
st.write("Tải lên một hình ảnh và AI sẽ tự động tạo ra một câu mô tả cho nó.")

# Tải các model cần thiết và hiển thị thông báo chờ
with st.spinner("Đang chuẩn bị mô hình, vui lòng chờ một chút..."):
    try:
        caption_model, tokenizer, feature_extractor = load_all_models()
        # Xác định độ dài tối đa của caption từ cấu trúc model
        # Input của model caption thường là [features, text_sequence]
        max_caption_length = caption_model.input_shape[1][1]
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.error("Vui lòng đảm bảo các tệp `best_model_with_attention.h5` và `tokenizer.pkl` nằm đúng trong thư mục chứa file `app.py`.")
        st.stop()


# Widget để người dùng tải ảnh lên
uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mở và hiển thị ảnh
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    st.write("") # Thêm một khoảng trống

    # Khi người dùng nhấn nút
    if st.button('Tạo chú thích! ✨'):
        with st.spinner("AI đang phân tích hình ảnh và suy nghĩ... 🤔"):
            # 1. Trích xuất đặc trưng từ ảnh
            features = extract_features(image, feature_extractor)
            
            # 2. Tạo chú thích từ đặc trưng
            caption = generate_caption(caption_model, tokenizer, features, max_caption_length)
            
            # 3. Hiển thị kết quả
            st.success("Hoàn thành!")
            st.subheader("Chú thích được tạo:")
            st.markdown(f"## *\"{caption}\"*")

st.markdown("---")
st.write("Xây dựng bởi Gemini | Model: ResNet50 + Attention (LSTM/GRU)")
