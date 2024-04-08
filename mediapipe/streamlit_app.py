import streamlit as st
from PIL import Image
import tensorflow as tf

# トレーニング済みモデルをロードする関数
def load_model(model_path='model/model.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 画像を前処理してモデルに渡すための関数
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, 0)
    return image_array

# 画像を分類する関数
def classify_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preprocessed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    return prediction

def main():
    st.title('MediaPipe Model Maker Image Classifier Demo')

    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_column_width=True)

        if st.button('画像を分類'):
            with st.spinner('分類中...'):
                interpreter = load_model()
                prediction = classify_image(image, interpreter)
                st.write('予測結果:', prediction)

if __name__ == '__main__':
    main()
