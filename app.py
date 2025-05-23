import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
from tensorflow.lite.python.interpreter import Interpreter
import tensorflow as tf
import numpy as np

def run_tflite_model(model_path):
    # تحميل موديل TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # الحصول على تفاصيل الإدخال والإخراج
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    # تجهيز بيانات عشوائية طبقاً لشكل الإدخال (يمكنك استبدالها ببيانات حقيقية)
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_data = np.array(np.random.random_sample(input_shape), dtype=input_dtype)

    # تعيين البيانات للإدخال
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # تشغيل التنبؤ
    interpreter.invoke()

    # استخراج النتائج من الإخراج
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output data:", output_data)

if __name__ == "__main__":
    # ضع هنا مسار ملف نموذج TFLite لديك
    model_file = "model.tflite"

    run_tflite_model(model_file)

# تحميل النموذج tflite من Google Drive إذا لم يكن موجودًا
MODEL_PATH = "model.tflite"
if not os.path.exists(MODEL_PATH):
    file_id = "1ECiRuPbY6m7gniTKIupGleiIuhgWdEce"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# تحميل النموذج باستخدام Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# واجهة Streamlit
st.set_page_config(page_title="Plant Diagnosis", layout="centered")
st.title("🌿 Plant Nutrient Diagnosis App")
st.write("Upload a plant leaf image to diagnose nutrient deficiency.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def process_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))  # تأكد من أن هذا هو الحجم المطلوب
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    try:
        processed = process_image(image)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        classes = ["Healthy Leaf", "Nitrogen Deficiency", "Zinc Deficiency"]
        result = classes[class_index]
        st.success(f"🧪 Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")