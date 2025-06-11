import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Layer
import tensorflow as tf

# --- PHẦN QUAN TRỌNG: ĐỊNH NGHĨA LỚP ATTENTION ---
# Keras không biết lớp 'Attention' là gì, chúng ta cần định nghĩa nó ở đây.
# Đây là một kiến trúc Attention phổ biến, rất có thể mô hình của bạn đã dùng nó.
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

# --- CÁC HÀM XỬ LÝ ---

# Hàm tải các model và tokenizer (được cache lại để chạy nhanh hơn)
@st.cache_resource
def load_all_models():
    # Khai báo lớp Attention là một custom object
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
