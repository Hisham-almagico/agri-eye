import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
import io
import tempfile
import os
import gdown
import zipfile

# Google Drive model ZIP ID
MODEL_PATH = "model.h5"
MODEL_ZIP = "model.zip"
MODEL_GDRIVE_ID = "1jJZ1ZexpJhvh8QUbH95-RgkJcyvBT5Uu"

if not os.path.exists(MODEL_PATH):
    st.warning("📥 Downloading model, please wait...")
    url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
    try:
        gdown.download(url, MODEL_ZIP, quiet=False)
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall()
        st.success("✅ Model downloaded and extracted successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download or extract the model: {e}")
        st.stop()

import traceback # Import traceback for detailed error logging

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Plant Health Diagnosis Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Configuration & Setup ---
MODEL_PATH = "model.h5"
# Define potential font paths (still needed for fallback check, but PDF forced to English)
ROBOTO_REGULAR_PATH = "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf"
ROBOTO_BOLD_PATH = "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf"
# ARABIC_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" # No longer needed for PDF

# --- Model Loading (Cached) ---
@st.cache_resource # Cache the model loading function
def load_plant_model(model_path):
    """Loads the Keras model using Streamlit's caching."""
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file 	'{model_path}'	 not found. Please ensure it's in the correct path.")
        return None
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"❌ Error loading the model from 	'{model_path}': {e}")
        st.text_area("Full Error Traceback:", traceback.format_exc(), height=200)
        return None

# Load the model using the cached function
model = load_plant_model(MODEL_PATH)

# Stop execution if model loading failed
if model is None:
    st.warning("⚠️ Application cannot proceed without a loaded model.")
    st.stop()

# --- Text & Translations (UI Only) ---
texts = {
    "English": {
        "page_title": "Plant Health Diagnosis Pro",
        "title": "🌿 Plant Nutrient Deficiency Diagnosis",
        "upload_label": "Upload Leaf Image",
        "upload_help": "Drag & drop or click to upload a clear image of a plant leaf (JPG, PNG). Limit 200MB.",
        "result": "Diagnosis Result:",
        "error": "❌ An error occurred:",
        "error_details": "Error Details:",
        "processing": "⏳ Analyzing image... Please wait.",
        "classes": ["Healthy Leaf", "Nitrogen Deficiency", "Zinc Deficiency"],
        "details": {
            "Healthy Leaf": {
                "en": "The leaf appears healthy with no significant signs of nutrient deficiency.",
                "ar": "تبدو الورقة سليمة ولا تظهر عليها علامات نقص العناصر الغذائية.",
                "fertilizer": "No specific fertilization required based on this diagnosis. Maintain standard care.",
                "fertilizer_ar": "لا يتطلب تسميدًا محددًا بناءً على هذا التشخيص. حافظ على الرعاية القياسية."
            },
            "Nitrogen Deficiency": {
                "en": ("Nitrogen deficiency typically causes yellowing of older leaves (chlorosis), starting from the tip, and stunted plant growth. "
                       "Consider using fertilizers high in nitrogen, such as urea or ammonium nitrate."),
                "ar": "يسبب نقص النيتروجين عادةً اصفرار الأوراق القديمة (الشحوب) بدءًا من الطرف، وتوقف نمو النبات. يُنصح باستخدام أسمدة غنية بالنيتروجين مثل اليوريا أو نترات الأمونيوم.",
                "fertilizer": "Recommended fertilizer: Urea (46% N) or Ammonium Nitrate (34% N). Follow product instructions carefully.",
                "fertilizer_ar": "الأسمدة الموصى بها: اليوريا (46% نيتروجين) أو نترات الأمونيوم (34% نيتروجين). اتبع تعليمات المنتج بعناية."
            },
            "Zinc Deficiency": {
                "en": ("Zinc deficiency often leads to smaller new leaves, yellow spots or bands between veins (interveinal chlorosis), and distorted leaf shapes. "
                       "Supplement with zinc-containing fertilizers like zinc sulfate, applied to soil or foliage."),
                "ar": "يؤدي نقص الزنك غالبًا إلى صغر حجم الأوراق الجديدة، وظهور بقع أو خطوط صفراء بين العروق (شحوب بيني)، وتشوه شكل الأوراق. يجب إضافة أسمدة تحتوي على الزنك مثل كبريتات الزنك، تطبق على التربة أو رشًا على الأوراق.",
                "fertilizer": "Recommended fertilizer: Zinc Sulfate (ZnSO4). Apply as per soil test results or foliar spray guidelines.",
                "fertilizer_ar": "الأسمدة الموصى بها: كبريتات الزنك (ZnSO4). طبق حسب نتائج تحليل التربة أو إرشادات الرش الورقي."
            }
        },
        "fertilizer_table_title": "General Fertilization Recommendations for Citrus (Example: Summer Orange)",
        "fertilizer_table": [
            ["Nutrient", "Recommended Amount (kg/ha/year)"],
            ["Nitrogen (N)", "150 - 200"],
            ["Phosphorus (P2O5)", "60 - 80"],
            ["Potassium (K2O)", "200 - 250"],
            ["Zinc (Zn)", "3 - 5 (Soil) or Foliar Spray"]
        ],
        "confidence": "Confidence Score",
        "download_pdf": "Download English Report (PDF)", # Label always English
        "report_title": "Plant Diagnosis Report", # PDF Title always English
        "uploaded_image": "Uploaded Image",
        "diagnosis": "Diagnosis",
        "recommendation": "Recommendation",
        "general_recommendations": "General Recommendations (Example)",
        "general_recommendations_expander": "View General Citrus Fertilization Guidelines"
    },
    "Arabic": {
        "page_title": "تشخيص صحة النبات برو",
        "title": "🌿 تشخيص نقص العناصر الغذائية في النبات",
        "upload_label": "رفع صورة الورقة",
        "upload_help": "اسحب وأفلت أو انقر لرفع صورة واضحة لورقة نبات (JPG, PNG). الحد الأقصى 200 ميجا بايت.",
        "result": "نتيجة التشخيص:",
        "error": "❌ حدث خطأ:",
        "error_details": "تفاصيل الخطأ:",
        "processing": "⏳ جارٍ تحليل الصورة... يرجى الانتظار.",
        "classes": ["ورقة سليمة", "نقص النيتروجين", "نقص الزنك"],
        "fertilizer_table_title": "توصيات تسميد عامة للحمضيات (مثال: البرتقال الصيفي)",
        "fertilizer_table": [
            ["العنصر الغذائي", "الكمية الموصى بها (كجم/هكتار/سنة)"],
            ["النيتروجين (N)", "150 - 200"],
            ["الفوسفور (P2O5)", "60 - 80"],
            ["البوتاسيوم (K2O)", "200 - 250"],
            ["الزنك (Zn)", "3 - 5 (تربة) أو رش ورقي"]
        ],
        "confidence": "درجة الثقة",
        "download_pdf": "تحميل التقرير الإنجليزي (PDF)", # Label always English
        "report_title": "Plant Diagnosis Report", # PDF Title always English
        "uploaded_image": "الصورة المرفوعة",
        "diagnosis": "التشخيص",
        "recommendation": "التوصية",
        "general_recommendations": "توصيات عامة (مثال)",
        "general_recommendations_expander": "عرض إرشادات تسميد الحمضيات العامة"
    }
}

# --- Enhanced CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Cairo:wght@400;700&display=swap');

body {
    font-family: 'Roboto', 'Cairo', sans-serif;
    background: linear-gradient(to bottom right, #dcedc8, #f1f8e9);
    color: #333;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1000px;
    margin: 1rem auto;
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(85, 139, 47, 0.15);
    border: 1px solid #c8e6c9;
}

/* Title styling */
h1 {
    text-align: center;
    color: #33691e; /* Darker green */
    margin-bottom: 1.5rem;
    font-family: 'Roboto', 'Cairo', sans-serif;
    font-weight: 700;
    font-size: 2.5rem;
    letter-spacing: 1px;
}

/* File uploader enhancement */
.stFileUploader {
    border: 2px dashed #81c784; /* Lighter green dash */
    border-radius: 10px;
    padding: 20px;
    background-color: #f1f8e9;
    transition: background-color 0.3s ease;
}
.stFileUploader:hover {
    background-color: #e8f5e9;
}
.stFileUploader label {
    font-size: 1.2rem;
    color: #33691e;
    font-weight: 600;
    text-align: center;
    display: block;
    margin-bottom: 10px;
}
.stFileUploader small {
    color: #558b2f;
    text-align: center;
    display: block;
}

/* Image display */
.stImage {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
    border: 1px solid #e0e0e0;
}

/* Result section styling */
.results-column {
    background-color: #f9fbe7; /* Very light yellow-green */
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e6ee9c;
}

/* Success message */
.stSuccess {
    background-color: #e8f5e9 !important; /* Light green */
    border: 1px solid #a5d6a7 !important;
    border-left: 5px solid #66bb6a !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    color: #2e7d32 !important;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* Confidence score */
.confidence-score {
    font-size: 1.1rem;
    color: #558b2f;
    margin-bottom: 1.5rem;
    text-align: center;
    font-weight: 500;
    background-color: #f1f8e9;
    padding: 8px;
    border-radius: 5px;
}

/* Details box */
.details-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 1rem;
    color: #424242;
    border: 1px solid #eeeeee;
    border-left: 5px solid #aed581; /* Light olive green */
    font-size: 1rem;
    line-height: 1.6;
}
.details-box b {
    color: #558b2f;
    font-weight: 700;
}

/* Expander for general recommendations */
.stExpander {
    border: 1px solid #dcedc8;
    border-radius: 10px;
    margin-top: 2rem;
    background-color: #ffffff;
}
.stExpander header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #33691e;
}

/* Table styling inside expander */
.stExpander .table-container h3 {
    color: #558b2f;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
    font-size: 1.3rem;
}
.stExpander table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}
.stExpander th, .stExpander td {
    border: 1px solid #dcedc8;
    padding: 12px 15px;
    text-align: center;
    font-size: 0.95rem;
    vertical-align: middle;
}
.stExpander th {
    background-color: #f1f8e9;
    color: #33691e;
    font-weight: 700;
}
.stExpander tr:nth-child(even) {
    background-color: #f9fbe7;
}

/* Download button */
.stDownloadButton button {
    width: 100%;
    background-color: #558b2f; /* Olive green */
    color: white;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    transition: background-color 0.3s ease, transform 0.1s ease;
    cursor: pointer;
    margin-top: 1rem;
}
.stDownloadButton button:hover {
    background-color: #33691e; /* Darker olive green */
    color: white;
    transform: translateY(-2px);
}
.stDownloadButton button:active {
    transform: translateY(0px);
}

/* Language selector */
.stSelectbox label {
    font-weight: bold;
    color: #33691e;
}

/* Footer caption */
.stCaption {
    text-align: center;
    margin-top: 3rem;
    color: #757575;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 {
        font-size: 2rem;
    }
    .stImage img {
        max-height: 350px;
        object-fit: contain;
    }
    .results-column {
        margin-top: 1.5rem;
    }
}

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def process_image(image_bytes):
    """Preprocesses the image bytes for the model."""
    try:
        image = Image.open(image_bytes).convert("RGB")
        image_resized = image.resize((256, 256))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return image, img_array
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.text_area("Image Processing Error Traceback:", traceback.format_exc(), height=150)
        return None, None

def create_pdf_report(pil_image, predicted_class_en, confidence):
    """Generates an English PDF report for the diagnosis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Font setup (Force English, use fallback if Roboto missing)
    roboto_regular_found = False
    roboto_bold_found = False
    default_font = "helvetica"

    try:
        if os.path.exists(ROBOTO_REGULAR_PATH):
            pdf.add_font("Roboto", "", ROBOTO_REGULAR_PATH, uni=True)
            roboto_regular_found = True
        if os.path.exists(ROBOTO_BOLD_PATH):
            pdf.add_font("Roboto", "B", ROBOTO_BOLD_PATH, uni=True)
            roboto_bold_found = True
    except Exception as font_e:
        # Silently ignore font loading errors, will use helvetica
        roboto_regular_found = roboto_bold_found = False

    # Determine fonts to use (Always English)
    english_font = "Roboto" if roboto_regular_found else default_font
    english_bold_font = "Roboto" if roboto_bold_found else default_font
    align = 'L'

    # --- PDF Content (Always English) ---
    pdf_texts = texts["English"] # Force English texts

    # Title
    pdf.set_font(english_bold_font, "B", 18)
    pdf.cell(0, 10, pdf_texts["report_title"], ln=True, align='C')
    pdf.ln(10)

    # Image
    pdf.set_font(english_bold_font, "B", 14)
    pdf.cell(0, 10, pdf_texts["uploaded_image"], ln=True, align=align)
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            pil_image.save(tmp_file, format="PNG")
            tmp_file_path = tmp_file.name

        img_w = 80
        page_w = pdf.w - 2 * pdf.l_margin
        img_x = (page_w - img_w) / 2 + pdf.l_margin
        pdf.image(tmp_file_path, x=img_x, w=img_w)
        pdf.ln(5)
    except Exception as e:
        pdf.set_font(english_font, "", 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Error embedding image: {e}", ln=True, align=align)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass # Silently ignore removal error

    # Diagnosis Result
    pdf.set_font(english_bold_font, "B", 14)
    pdf.cell(0, 10, pdf_texts["diagnosis"], ln=True, align=align)
    pdf.set_font(english_font, "", 12)
    pred_class_display = predicted_class_en # Already English
    conf_label = pdf_texts["confidence"]
    # Use multi_cell for result as it might wrap
    pdf.multi_cell(0, 8, f"{pdf_texts['result']} {pred_class_display}", align=align)
    # Use cell for confidence score - split label and value if needed
    try:
        # Attempt to write confidence score on one line using cell
        pdf.cell(0, 8, f"{conf_label}: {confidence:.2f}%", ln=True, align=align)
    except Exception as cell_error:
        # Fallback: Write label and value on separate lines if the first attempt fails
        st.warning(f"FPDF cell error for confidence score: {cell_error}. Using fallback layout.")
        pdf.cell(0, 8, f"{conf_label}:", ln=True, align=align)
        pdf.cell(0, 8, f"{confidence:.2f}%", ln=True, align=align)
    pdf.ln(5)

    # Recommendation Details
    pdf.set_font(english_bold_font, "B", 14)
    pdf.cell(0, 10, pdf_texts["recommendation"], ln=True, align=align)
    pdf.set_font(english_font, "", 12)
    details_text = texts["English"]["details"].get(predicted_class_en, {}).get("en", "N/A")
    fertilizer_text = texts["English"]["details"].get(predicted_class_en, {}).get("fertilizer", "N/A")
    pdf.multi_cell(0, 8, details_text, align=align)
    pdf.ln(3)
    pdf.set_font(english_bold_font, "B", 12)
    pdf.multi_cell(0, 8, fertilizer_text, align=align)
    pdf.ln(10)

    # General Fertilization Table
    pdf.set_font(english_bold_font, "B", 14)
    pdf.cell(0, 10, pdf_texts["general_recommendations"], ln=True, align=align)
    pdf.set_font(english_bold_font, "B", 12)
    pdf.cell(0, 10, pdf_texts["fertilizer_table_title"], ln=True, align='C')

    pdf.set_font(english_font, "", 10)
    table_data = pdf_texts['fertilizer_table']
    headers = table_data[0]
    data_rows = table_data[1:]

    col_width = (pdf.w - 2 * pdf.l_margin) / len(headers)
    line_height = pdf.font_size * 1.8

    pdf.set_fill_color(224, 242, 241) # Light teal background for header
    pdf.set_font(english_bold_font, "B", 11)
    for header in headers:
        pdf.cell(col_width, line_height, header, border=1, align='C', fill=True)
    pdf.ln(line_height)

    pdf.set_font(english_font, "", 10)
    pdf.set_fill_color(255, 255, 255)
    fill = False
    for row in data_rows:
        for item in row:
            pdf.cell(col_width, line_height, item, border=1, align='C', fill=fill)
        pdf.ln(line_height)
        fill = not fill # Alternate row fill slightly

    try:
        # Corrected: pdf.output returns bytes (bytearray), no need to encode again
        pdf_output_bytes = pdf.output(dest='S')
        return io.BytesIO(pdf_output_bytes)
    except Exception as pdf_e:
        st.error(f"❌ Error generating PDF output: {pdf_e}")
        st.text_area("PDF Generation Error Traceback:", traceback.format_exc(), height=150)
        return None

# --- Streamlit App Layout ---

# Language Selection (UI Only)
col1_lang, col2_lang = st.columns([1, 8]) # Adjust column ratio
with col1_lang:
    lang_options = list(texts.keys())
    selected_lang = st.selectbox("🌐 Language", lang_options, label_visibility="collapsed")

if 'lang' not in st.session_state or st.session_state.lang != selected_lang:
    st.session_state.lang = selected_lang

current_texts = texts[st.session_state.lang]

st.markdown(f"<h1>{current_texts['title']}</h1>", unsafe_allow_html=True)

# File Uploader Section
with st.container(): # Use container for better grouping
    uploaded_file = st.file_uploader(
        current_texts['upload_label'],
        type=["jpg", "jpeg", "png"],
        help=current_texts['upload_help'],
        label_visibility="visible"
    )

if uploaded_file is not None:
    file_bytes = io.BytesIO(uploaded_file.getvalue())

    # Process Image and Predict
    pil_image, processed_image = process_image(file_bytes)

    if processed_image is not None and pil_image is not None:
        col1_img, col2_results = st.columns([2, 3]) # Adjust column ratio

        with col1_img:
            st.image(pil_image, caption=current_texts["uploaded_image"], use_container_width=True)

        with col2_results:
            with st.container(): # Container for results section styling
                st.markdown('<div class="results-column">', unsafe_allow_html=True)
                processing_placeholder = st.empty()
                processing_placeholder.info(current_texts['processing'])
                try:
                    if model is None:
                         st.error("❌ Model not loaded. Cannot perform prediction.")
                    else:
                        prediction = model.predict(processed_image)[0]
                        processing_placeholder.empty()

                        class_index = np.argmax(prediction)
                        confidence = float(np.max(prediction)) * 100
                        pred_class_en = texts["English"]["classes"][class_index]
                        # Get display class name based on UI language
                        pred_class_display = current_texts["classes"][class_index]

                        st.success(f"{current_texts['result']} {pred_class_display}")
                        st.markdown(f"<p class='confidence-score'>{current_texts['confidence']}: {confidence:.2f}%</p>", unsafe_allow_html=True)

                        # Get detail/fertilizer text based on UI language
                        detail_key = "ar" if st.session_state.lang == "Arabic" else "en"
                        fertilizer_key = "fertilizer_ar" if st.session_state.lang == "Arabic" else "fertilizer"
                        details_text = texts["English"]["details"].get(pred_class_en, {}).get(detail_key, "N/A")
                        fertilizer_text = texts["English"]["details"].get(pred_class_en, {}).get(fertilizer_key, "N/A")

                        st.markdown(f"<div class='details-box'>{details_text}<br><br><b>{current_texts['recommendation']}:</b> {fertilizer_text}</div>", unsafe_allow_html=True)

                        # PDF Generation (Always English)
                        pdf_file = create_pdf_report(pil_image, pred_class_en, confidence)
                        if pdf_file:
                            st.download_button(
                                label=f"📄 {current_texts['download_pdf']}", # Uses UI lang text for button
                                data=pdf_file,
                                file_name="plant_diagnosis_report_english.pdf", # Always English filename
                                mime="application/pdf",
                                key="download-pdf-button"
                            )
                        else:
                            st.warning("⚠️ Could not generate PDF report.")

                except Exception as e:
                    processing_placeholder.empty()
                    st.error(f"{current_texts['error']}")
                    st.text_area(f"{current_texts['error_details']}", f"{e}\n\n{traceback.format_exc()}", height=200)
                st.markdown('</div>', unsafe_allow_html=True) # Close results-column div
    else:
        st.warning("⚠️ Image processing failed. Cannot proceed with diagnosis.")

    # Display General Fertilization Table in an Expander
    with st.expander(current_texts["general_recommendations_expander"]):
        st.markdown(f"<div class='table-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{current_texts['fertilizer_table_title']}</h3>", unsafe_allow_html=True)
        table_data = current_texts['fertilizer_table']

        table_html = "<table><thead><tr>"
        # Reverse headers for Arabic UI
        effective_headers = reversed(table_data[0]) if st.session_state.lang == "Arabic" else table_data[0]
        for header in effective_headers:
            table_html += f"<th>{header}</th>"
        table_html += "</tr></thead><tbody>"
        for row in table_data[1:]:
            table_html += "<tr>"
            # Reverse row cells for Arabic UI
            effective_row = reversed(row) if st.session_state.lang == "Arabic" else row
            for cell in effective_row:
                table_html += f"<td>{cell}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
