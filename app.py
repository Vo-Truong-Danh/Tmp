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
import io # Thêm import io để xử lý dữ liệu ảnh từ streamlit

# --- Thiết lập trang ---
st.set_page_config(page_title="Trình tạo Chú thích Ảnh", layout="centered")

# --- Định nghĩa lớp Attention tùy chỉnh ---
# Lớp này CẦN PHẢI KHỚP với lớp Attention đã được sử dụng khi bạn huấn luyện mô hình.
# Nếu không khớp, hoặc nếu mô hình của bạn không sử dụng lớp Attention, có thể cần điều chỉnh.
# Lỗi "Unrecognized keyword arguments passed to LSTM: {'time_major': False}"
# thường gợi ý rằng Keras cần biết cách tải lớp tùy chỉnh này.
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape == (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    # get_config là cần thiết để Keras có thể lưu và tải lớp tùy chỉnh
    def get_config(self):
        config = super(Attention, self).get_config()
        # Bạn cần thêm các đối số của hàm __init__ vào đây nếu có
        # Ví dụ: config.update({'units': self.units})
        # Đối với Attention đơn giản này, không có đối số nào cần lưu trong config
        return config


# --- Tải Model và Tokenizer (Sử dụng cache để không tải lại mỗi lần) ---
@st.cache_resource
def load_all_models():
    """
    Tải tất cả các model cần thiết: model tạo caption, tokenizer, và model trích xuất đặc trưng.
    Hàm này được cache để tăng tốc độ.
    """
    with st.spinner('Đang chuẩn bị mô hình, vui lòng chờ một chút...'):
        try:
            # Tải model tạo caption đã được huấn luyện
            # Đảm bảo tệp 'best_model_with_attention.h5' nằm cùng thư mục
            # custom_objects cần thiết để tải lớp Attention
            custom_objects = {'Attention': Attention}
            caption_model = load_model('best_model_with_attention.h5', custom_objects=custom_objects)
            st.success("Mô hình tạo chú thích đã được tải thành công!")

            # Tải tokenizer
            # Đảm bảo tệp 'tokenizer.pkl' nằm cùng thư mục
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            st.success("Tokenizer đã được tải thành công!")

            # Tải model ResNet50 để trích xuất đặc trưng ảnh
            # Chúng ta chỉ lấy các lớp đến trước lớp phân loại cuối cùng
            image_model = ResNet50(weights='imagenet')
            # Tạo một model mới bằng cách lấy đầu ra của lớp áp chót (thường là GlobalAveragePooling2D)
            feature_extractor = tf.keras.Model(inputs=image_model.inputs, outputs=image_model.layers[-2].output)
            st.success("Mô hình trích xuất đặc trưng (ResNet50) đã được tải thành công!")

            return caption_model, tokenizer, feature_extractor
        except FileNotFoundError:
            st.error(
                f"Lỗi: Không tìm thấy tệp mô hình hoặc tokenizer. "
                f"Vui lòng đảm bảo các tệp **best_model_with_attention.h5** và **tokenizer.pkl** "
                f"nằm đúng trong thư mục chứa file **app.py**."
            )
            st.stop() # Dừng ứng dụng nếu không tìm thấy tệp
        except Exception as e:
            # Xử lý lỗi cụ thể nếu liên quan đến đối số 'time_major'
            if "Unrecognized keyword arguments passed to LSTM: {'time_major': False}" in str(e):
                st.error(
                    f"Lỗi khi tải model: {e}"
                    f"\n\nĐây thường là lỗi không tương thích phiên bản TensorFlow/Keras hoặc lớp Attention. "
                    f"Mô hình được huấn luyện với một phiên bản Keras/TensorFlow khác hoặc một định nghĩa lớp Attention khác. "
                    f"Bạn có thể cần thay đổi phiên bản TensorFlow trong requirements.txt "
                    f"(ví dụ: thử `tensorflow==2.11.0` hoặc `tensorflow==2.12.0`)."
                    f"\nVui lòng kiểm tra lại định nghĩa lớp `Attention` trong file này và so sánh với mô hình gốc."
                )
            else:
                st.error(f"Lỗi khi tải model hoặc tokenizer: {e}")
            st.stop() # Dừng ứng dụng nếu có lỗi tải
    return None, None, None # Trả về None nếu có lỗi

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
    # Sử dụng .get() để tránh lỗi nếu không tìm thấy index
    return tokenizer.index_word.get(integer)

# --- Hàm tạo Caption (Sử dụng thuật toán Greedy Search) ---
def generate_caption(model, tokenizer, photo_features, max_length):
    """
    Tạo ra một câu chú thích từ đặc trưng ảnh.
    """
    # Bắt đầu chuỗi với token 'startseq'
    in_text = 'startseq'
    # Lặp lại để tạo từng từ trong câu
    for i in range(max_length): # Sử dụng i để tránh trùng lặp với max_length
        # Chuyển chuỗi hiện tại thành dạng số
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Padding để chuỗi có độ dài cố định
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Dự đoán từ tiếp theo
        # Model nhận 2 đầu vào: features ảnh và chuỗi văn bản hiện tại
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
    # Loại bỏ 'startseq' và 'endseq'
    if 'startseq' in final_caption:
        final_caption.remove('startseq')
    if 'endseq' in final_caption:
        final_caption.remove('endseq')

    final_caption = ' '.join(final_caption)
    # Viết hoa chữ cái đầu và thêm dấu chấm
    return final_caption.capitalize() + '.'

# --- Giao diện người dùng Streamlit ---

st.title("📷 Trình tạo Chú thích Ảnh bằng AI")
st.write("Tải lên một hình ảnh và AI sẽ tự động tạo ra một câu mô tả cho nó.")

# Tải các model cần thiết và hiển thị thông báo chờ
caption_model, tokenizer, feature_extractor = load_all_models()

# Xác định độ dài tối đa của caption từ cấu trúc model
# Input của model caption thường là [features, text_sequence]
# Giả sử input thứ hai (index 1) là input cho text sequence, và chiều thứ hai (index 1) của nó là max_length
try:
    max_caption_length = caption_model.input_shape[1][1]
    st.info(f"Độ dài chú thích tối đa được hỗ trợ: {max_caption_length} từ.")
except Exception as e:
    st.warning(f"Không thể xác định độ dài chú thích tối đa từ mô hình. Sử dụng giá trị mặc định (34). Lỗi: {e}")
    max_caption_length = 34 # Giá trị mặc định nếu không xác định được


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

