import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# --- CÁC HÀM TẢI MÔ HÌNH VÀ TOKENIZER (Sử dụng cache để tối ưu) ---

@st.cache_resource
def load_models_and_tokenizer():
    """Tải mô hình tạo chú thích, mô hình ResNet50 để trích xuất đặc trưng và tokenizer."""
    
    # Tải mô hình chính
    try:
        caption_model = load_model('best_model_with_attention.h5')
    except Exception as e:
        st.error(f"Lỗi khi tải file 'best_model_with_attention.h5': {e}")
        st.error("Vui lòng đảm bảo file mô hình đã được tải về và đặt đúng trong thư mục.")
        return None, None, None

    # Tải tokenizer
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi khi tải file 'tokenizer.pkl': {e}")
        st.error("Vui lòng đảm bảo file tokenizer đã được tải về và đặt đúng trong thư mục.")
        return None, None, None
        
    # Tải mô hình ResNet50 để trích xuất đặc trưng
    resnet_model = ResNet50(weights='imagenet')
    feature_extractor = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    
    return caption_model, feature_extractor, tokenizer

# --- CÁC HÀM XỬ LÝ (Lấy từ notebook) ---

def extract_feature_from_image(feature_extractor, image):
    """Trích xuất vector đặc trưng từ một ảnh."""
    image = image.resize((224, 224))
    image = np.array(image)
    
    # Chuyển ảnh RGB 3 kênh thành 4 chiều để tương thích với mô hình
    if image.shape[2] == 4: # Xử lý ảnh có kênh alpha (RGBA)
        image = image[..., :3]
        
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    feature = feature_extractor.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    """Chuyển index thành từ."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_feature, tokenizer, max_length):
    """Tạo chú thích cho ảnh từ vector đặc trưng."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([image_feature, sequence], verbose=0)
        
        pred_idx = np.argmax(yhat)
        word = idx_to_word(pred_idx, tokenizer)
        
        if word is None:
            break
        
        in_text += ' ' + word
        
        if word == 'endseq':
            break
            
    return in_text

# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---

st.set_page_config(page_title="Tạo Chú Thích Ảnh Tiếng Việt", layout="centered")

st.title("️🖼️ Tạo Chú Thích Ảnh Tiếng Việt")
st.write("Tải lên một hình ảnh và mô hình sẽ tự động tạo ra một chú thích bằng tiếng Việt mô tả nội dung của ảnh đó.")
st.write("---")

# Tải các mô hình và tokenizer
caption_model, feature_extractor, tokenizer = load_models_and_tokenizer()

# Kiểm tra nếu tải mô hình thành công
if caption_model and feature_extractor and tokenizer:
    uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh đã tải lên
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
        
        st.write("")
        
        # Tạo chú thích khi nhấn nút
        if st.button('Tạo chú thích'):
            with st.spinner('Đang xử lý và tạo chú thích...'):
                max_length = 41 # Giá trị này được xác định trong notebook
                
                # Trích xuất đặc trưng ảnh
                photo_feature = extract_feature_from_image(feature_extractor, image)
                
                # Tạo chú thích
                generated_caption = predict_caption(caption_model, photo_feature, tokenizer, max_length)
                
                # Xử lý chuỗi kết quả cho đẹp
                final_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip().capitalize()
                
                st.success('Đã tạo xong!')
                st.subheader('**Chú thích được tạo:**')
                st.write(f"### *{final_caption}*")
else:
    st.warning("Ứng dụng chưa sẵn sàng do không tải được mô hình hoặc tokenizer. Vui lòng kiểm tra lại file.")