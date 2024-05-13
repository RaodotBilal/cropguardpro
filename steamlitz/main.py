import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from io import BytesIO
import base64

# Load the model
MODEL = tf.keras.models.load_model("model.keras")
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

# Function to read file as image
def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file.read())))
    return image


# Streamlit app
def main():
    st.title("Image Classifier")

    # Sidebar to upload file
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_file is not None:
        image = read_file_as_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Classify image on button click
        if st.sidebar.button("Classify"):
            # Process image
            img_batch = np.expand_dims(image, 0)

            # Make predictions
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            # Display result
            st.write(f"Class: {predicted_class}")
            st.write(f"Confidence: {confidence}")

@st.experimental_singleton
def classify_image(encoded_image):
    image = read_file_as_image(base64.b64decode(encoded_image))
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

if __name__ == "__main__":
    main()