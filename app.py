import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
from metrics import mcc_loss, mcc_metric, dice_coef, dice_loss, f1, tversky, tversky_loss, focal_tversky_loss, bce_dice_loss_new, jaccard

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Alternatively, set a memory limit
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # Set limit in MB
    except RuntimeError as e:
        print(e)


# Define custom objects
custom_objects = {
    'Addons>GroupNormalization': tfa.layers.GroupNormalization,
    'mcc_loss': mcc_loss,
    'mcc_metric': mcc_metric,
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'f1': f1,
    'tversky': tversky,
    'tversky_loss': tversky_loss,
    'focal_tversky_loss': focal_tversky_loss,
    'bce_dice_loss_new': bce_dice_loss_new,
    'jaccard': jaccard
}



# Load models with custom objects
sandboil_model = load_model('sandboil_best_model.h5', custom_objects=custom_objects)
seepage_model = load_model('seepage_best_model.h5', custom_objects=custom_objects)

# UI Setup for Streamlit
st.title("Sandboil and Seepage Detection")

# Checkboxes for model selection
sandboil_selected = st.checkbox("Detect Sandboils")
seepage_selected = st.checkbox("Detect Seepage")

# Function to process the image
input_width, input_height = 256, 256  # Adjust based on your model's input size

def process_image(image):
    image = cv2.resize(image, (input_width, input_height))  # Resize the image
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to apply the model
from tensorflow.keras import backend as K

def apply_model(selected_model, image):
    processed_image = process_image(image)
    predictions = selected_model.predict(processed_image)
    K.clear_session()  # Clear session to free up memory
    return predictions

# Detect sandboil
if sandboil_selected:
    st.write("Selected Model: Sandboil Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png'])
    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        predictions = apply_model(sandboil_model, image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Predictions: ', predictions)  # Adjust visualization as needed

# Detect seepage
if seepage_selected:
    st.write("Selected Model: Seepage Detection")
    uploaded_image = st.file_uploader("Upload an Image for Seepage Detection", type=['jpg', 'png'])
    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        predictions = apply_model(seepage_model, image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Predictions: ', predictions)  # Adjust visualization as needed

