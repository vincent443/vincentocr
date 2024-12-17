# -*- coding: utf-8 -*-


import streamlit as st
import cv2
import pytesseract
import pandas as pd
from PIL import Image
import tempfile
import os

# Set Tesseract OCR path (handled automatically if hosted)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image):
    """
    Preprocess the image to improve OCR results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def perform_ocr(image):
    """
    Extract text from the preprocessed image using Tesseract OCR.
    """
    text = pytesseract.image_to_string(image, config="--psm 6")
    return text

def parse_text_to_dataframe(text):
    """
    Parse OCR output text into structured rows and columns.
    """
    data = []
    lines = text.split("\n")
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                data.append([parts[0], parts[1], parts[2], parts[3]])
            elif len(parts) > 1:
                data.append(parts + [""] * (4 - len(parts)))
    return pd.DataFrame(data, columns=["ID", "Host", "Time In", "Time Out"])

# Streamlit App
st.title("Visitors Log OCR Tool")

uploaded_files = st.file_uploader("Upload Images of Visitor Logs", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    all_data = []

    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Read and preprocess image
        image = Image.open(uploaded_file)
        image = preprocess_image(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        # Perform OCR
        text = perform_ocr(image)
        st.text_area("Extracted Text", text, height=200)

        # Parse data into DataFrame
        df = parse_text_to_dataframe(text)
        st.write("Parsed Data", df)
        all_data.append(df)

    # Combine all data and save to Excel
    if st.button("Download Excel"):
        combined_df = pd.concat(all_data, ignore_index=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            combined_df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            st.download_button(
                label="Download Visitor Log as Excel",
                data=tmp.read(),
                file_name="visitor_log.xlsx",
                mime="application/vnd.ms-excel"
            )
