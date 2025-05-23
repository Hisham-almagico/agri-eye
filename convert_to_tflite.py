import tensorflow as tf

# تحميل النموذج
model = tf.keras.models.load_model("model.h5")

# تحويل النموذج إلى TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# حفظ النموذج الجديد
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ تم تحويل النموذج إلى model.tflite بنجاح.")