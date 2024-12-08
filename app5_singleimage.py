import streamlit as st
import cv2
import subprocess
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import backend as K
from metrics import mcc_loss, mcc_metric, dice_coef, dice_loss, f1, tversky, tversky_loss, focal_tversky_loss, bce_dice_loss_new, jaccard, bce_dice_loss
from SandBoilNet import PCALayer, spatial_pooling_block, attention_block, initial_conv2d_bn, conv2d_bn, iterLBlock, decoder_block
# SandboilNet_Dropout, old_attention_block
import gc
import os
import time
from io import BytesIO
from PIL import Image
import zipfile


# #--- GPU Management ---
def kill_gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE)
    pids = result.stdout.decode('utf-8').strip().split('\n')
    for pid in pids:
        if pid.isdigit():
            try:
                os.kill(int(pid), 9)
                print(f"Killed process with PID: {pid}")
            except Exception as e:
                print(f"Could not kill process {pid}: {e}")
#kill_gpu_processes()

def enable_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs.")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")

enable_memory_growth()

def clear_tf_memory():
    K.clear_session()
    gc.collect()
    print("Cleared TensorFlow session and garbage collected.")




# # Function to save uploaded files to a temporary directory
# def save_uploaded_file(uploaded_file, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     file_path = os.path.join(save_dir, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return file_path

# # Function to overlay an image with a sample overlay (e.g., a semi-transparent rectangle)
# def overlay_image(image):
#     overlay = image.copy()
#     height, width = image.shape[:2]
#     # Example: Draw a semi-transparent rectangle in the center of the image
#     start_x, start_y = width // 4, height // 4
#     end_x, end_y = 3 * width // 4, 3 * height // 4
#     cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), -1)  # Green rectangle
#     alpha = 0.5  # Transparency factor
#     overlaid_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#     return overlaid_image

# # Function to create a ZIP file from a folder of images
# def create_zip(folder_path):
#     zip_buffer = BytesIO()
#     with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 arcname = os.path.relpath(file_path, folder_path)  # Relative path for ZIP archive
#                 zipf.write(file_path, arcname)
#     zip_buffer.seek(0)
#     return zip_buffer

# # Streamlit App UI
# st.title("Individual Image Overlay and ZIP Download")

# # Step 1: Upload multiple image files
# uploaded_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# if uploaded_files:
#     temp_dir = "temp_images"
#     output_dir = "output_images"
    
#     # Save uploaded files to a temporary directory
#     for uploaded_file in uploaded_files:
#         save_uploaded_file(uploaded_file, temp_dir)
    
#     st.success(f"Uploaded {len(uploaded_files)} files to {temp_dir}.")
    
#     # Step 2: Process each image and save the overlaid result
#     os.makedirs(output_dir, exist_ok=True)
    
#     for file_name in os.listdir(temp_dir):
#         input_path = os.path.join(temp_dir, file_name)
#         output_path = os.path.join(output_dir, file_name)
        
#         # Read the image using OpenCV
#         image = cv2.imread(input_path)
#         if image is not None:
#             # Apply overlay to the image
#             overlaid_image = overlay_image(image)
#             # Save the overlaid image to the output directory
#             cv2.imwrite(output_path, overlaid_image)
    
#     st.success(f"Processed {len(os.listdir(temp_dir))} images and saved them to {output_dir}.")
    
#     # Step 3: Create ZIP file with all overlaid images
#     if st.button("Download Overlaid Images as ZIP"):
#         zip_buffer = create_zip(output_dir)
        
#         # Step 4: Provide download button for ZIP file
#         st.download_button(
#             label="Download ZIP File",
#             data=zip_buffer,
#             file_name="overlaid_images.zip",
#             mime="application/zip"
#         )










# Define custom objects for loading models
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
    'jaccard': jaccard,
    'PCALayer': PCALayer,
    'bce_dice_loss': bce_dice_loss
}


@st.cache_resource
def load_sandboil_model():
    return load_model('sandboil_best_model.h5', custom_objects=custom_objects)

@st.cache_resource
def load_seepage_model():
    return load_model('seepage_best_model.h5', custom_objects=custom_objects)

# Allow user to choose between image or video processing
processing_choice = st.radio("Choose the type of input you want to process:", ("Image", "Video"))
# Allow user to choose between overlay or bounding box detection
detection_choice = st.radio("Choose the type of detection:", ("Overlay", "Bounding Box"))

# UI Setup for Streamlit
st.title("Sandboil and Seepage Detection - Image and Video")

# Checkboxes for model selection
sandboil_selected = st.checkbox("Detect Sandboils")
seepage_selected = st.checkbox("Detect Seepage")
crack_selected = st.checkbox("Detect Crack")
potholes_selected = st.checkbox("Detect Potholes")
encroachment_selected = st.checkbox("Detect Encroachment")
rutting_selected = st.checkbox("Detect Rutting")
animal_burrow_selected = st.checkbox("Detect Animal Burrow")
vegetation_selected = st.checkbox("Detect Vegetation")


# Sidebar legend for detection types and their corresponding colors
@st.cache_data
def render_legend():
    legend_html = """
    <span style='color:green'>■ Sandboils</span><br>
    <span style='color:pink'>■ Seepage</span><br>
    <span style='color:blue'>■ Crack</span><br>
    <span style='color:orange'>■ Potholes</span><br>
    <span style='color:red'>■ Encroachment</span><br>
    <span style='color:purple'>■ Rutting</span><br>
    <span style='color:brown'>■ Animal Burrow</span><br>
    <span style='color:darkgreen'>■ Vegetation</span>
    """
    return legend_html

st.sidebar.write("### Legend")
st.sidebar.markdown(render_legend(), unsafe_allow_html=True)

# Optional: Add logic to highlight the selected items in the main content area
if sandboil_selected:
    st.write("Sandboils detection is selected.")
if seepage_selected:
    st.write("Seepage detection is selected.")
if crack_selected:
    st.write("Crack detection is selected.")
if potholes_selected:
    st.write("Potholes detection is selected.")
if encroachment_selected:
    st.write("Encroachment detection is selected.")
if rutting_selected:
    st.write("Rutting detection is selected.")
if animal_burrow_selected:
    st.write("Animal Burrow detection is selected.")
if vegetation_selected:
    st.write("Vegetation detection is selected.")
        
@st.cache_data
def preprocess_image(image, model_type, resolution_factor=1.0, brightness_factor=0, contrast_factor=0,
                     blur_amount=1, edge_detection=False, flip_horizontal=False, flip_vertical=False,
                     rotate_angle=0):
    """
    Preprocess the image based on the selected model type and additional transformations.
    """
    # Set input dimensions dynamically based on the model type
    if model_type == "sandboil":
        input_width, input_height = 512, 512  # Sandboil model dimensions
    elif model_type == "seepage":
        input_width, input_height = 256, 256  # Seepage model dimensions
    else:
        raise ValueError("Invalid model type. Choose 'sandboil' or 'seepage'.")

    # Resize image to match the model's expected input size
    image_resized = cv2.resize(image, (input_width, input_height))

    # Apply resolution scaling
    new_width = int(image.shape[1] * resolution_factor)
    new_height = int(image.shape[0] * resolution_factor)
    image = cv2.resize(image, (new_width, new_height))

    # Brightness and contrast adjustments
    if brightness_factor != 0 or contrast_factor != 0:
        image = cv2.convertScaleAbs(image, alpha=1 + contrast_factor / 100.0, beta=brightness_factor)

    # Gaussian blur
    if blur_amount > 1:
        image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Edge detection
    if edge_detection:
        image = cv2.Canny(image, threshold1=100, threshold2=200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

    # Flip horizontally or vertically
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)

    # Rotate the image
    if rotate_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    # Normalize and add batch dimension
    return np.expand_dims(image_resized / 255.0, axis=0)

def process_frame(frame, input_width, input_height):
    """Process a single frame."""
    frame_resized = cv2.resize(frame, (input_width, input_height))
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    K.clear_session()
    return np.expand_dims(frame_normalized, axis=0)

def apply_model_to_frame(model, frame, input_width, input_height):
    """Apply the model to a single frame."""
    processed_frame = process_frame(frame, input_width, input_height)
    predictions = model.predict(processed_frame)
    predicted_mask = np.squeeze(predictions) 
    K.clear_session()
    return predicted_mask


def overlay_mask_on_frame(frame, mask, alpha=0.5, color=(0, 255, 0)):
    """Overlay the segmentation mask on the original frame."""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_colored = np.stack([mask_resized] * 3, axis=-1) if len(mask_resized.shape) == 2 else mask_resized
    mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])
    overlaid_frame = cv2.addWeighted(frame.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    K.clear_session()
    return overlaid_frame


def draw_bounding_boxes(frame, mask):
    """Draw bounding boxes around detected regions."""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)  # Green bounding box
    K.clear_session()
    return frame 



# Sidebar form for sliders
with st.sidebar.form("slider_form"):
    st.write("### Customize Image Processing")

    # Sliders for image processing customization
    resolution_factor = st.slider("Adjust Image Resolution Scaling Factor", 0.1, 2.0, 1.0)
    brightness_factor = st.slider("Adjust Brightness", -100, 100, 0)
    contrast_factor = st.slider("Adjust Contrast", -100, 100, 0)
    blur_amount = st.slider("Apply Gaussian Blur (Kernel Size)", 1, 15, step=2)
    edge_detection = st.checkbox("Apply Edge Detection (Canny)")
    flip_horizontal = st.checkbox("Flip Horizontally")
    flip_vertical = st.checkbox("Flip Vertically")
    rotate_angle = st.slider("Rotate Image (Degrees)", -180, 180, step=1)

    # Submit button
    submitted = st.form_submit_button("Submit")


def apply_model(selected_model, image, model_type):
    """
    Apply the selected model to a preprocessed image.
    """
    # Preprocess the image based on the model type
    processed_image = preprocess_image(
        image=image.copy(),
        model_type=model_type,
        resolution_factor=resolution_factor,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        blur_amount=blur_amount,
        edge_detection=edge_detection,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        rotate_angle=rotate_angle
    )
    
    # Predict using the selected model
    predictions = selected_model.predict(processed_image)
    # Remove batch and channel dimensions if necessary
    predicted_mask = np.squeeze(predictions)
    # Clear session and invoke garbage collection to free up GPU memory
    K.clear_session()
    gc.collect()
    return predicted_mask




# Function to overlay mask on image with color and intensity options
def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask is in the same size as the image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Convert grayscale mask to color if needed (e.g., binary or single-channel)
    if len(mask_resized.shape) == 2:  # If it's a single-channel grayscale or binary mask
        mask_colored = np.stack([mask_resized] * 3, axis=-1)  # Convert to 3-channel RGB
        
        # Apply custom color based on detection type
        mask_colored = np.where(mask_colored > 0.5, color, [0, 0, 0])  # Use selected color
    
    else:
        mask_colored = mask_resized
    
    # Blend images using addWeighted
    overlaid_image = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, mask_colored.astype(np.uint8), alpha, 0)
    K.clear_session()
    return overlaid_image

# Slider to control overlay intensity (transparency)
overlay_intensity = st.sidebar.slider("Overlay Intensity", 0.0, 1.0, 0.5)


if processing_choice == "Image":
    # Image Upload Section
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png'])
    if uploaded_image is not None:
        # Read and decode uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image[..., ::-1], caption="Uploaded Image", use_column_width=True)

        # Initialize combined overlay
        combined_overlay = np.zeros_like(image, dtype=np.uint8)

        # Load both models if both checkboxes are selected
        sandboil_model = None
        seepage_model = None

        if sandboil_selected:
            st.write("Loading Sandboil model...")
            sandboil_model = load_sandboil_model()

        if seepage_selected:
            st.write("Loading Seepage model...")
            seepage_model = load_seepage_model()

        # Run Sandboil Detection if selected
        if sandboil_selected and sandboil_model is not None:
            st.write("Running Sandboil Detection...")
            sandboil_predictions = apply_model(sandboil_model, image, model_type="sandboil")
            sandboil_overlay = overlay_mask_on_image(
                image, sandboil_predictions, alpha=overlay_intensity, color=(0, 255, 0)
            )
            combined_overlay = cv2.addWeighted(combined_overlay, 1.0, sandboil_overlay, 1.0, 0)

        # Run Seepage Detection if selected
        if seepage_selected and seepage_model is not None:
            st.write("Running Seepage Detection...")
            seepage_predictions = apply_model(seepage_model, image, model_type="seepage")
            seepage_overlay = overlay_mask_on_image(
                image, seepage_predictions, alpha=overlay_intensity, color=(255, 105, 180)
            )
            combined_overlay = cv2.addWeighted(combined_overlay, 1.0, seepage_overlay, 1.0, 0)

        # Display the combined overlay
        st.image(combined_overlay[..., ::-1], caption='Combined Detection Overlay', use_column_width=True)
    else:
        st.warning("Please upload an image to proceed.")
        
        
elif processing_choice == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(temp_file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = "output_video.mp4"
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        stframe = st.empty()
        
        # Dynamically set dimensions based on selected model
        if sandboil_selected:
            input_width, input_height = 512, 512
            model = load_sandboil_model()
        elif seepage_selected:
            input_width, input_height = 256, 256
            model = load_seepage_model()
        else:
            st.warning("Please select a detection model.")
            cap.release()
            out.release()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            predictions = apply_model_to_frame(model, frame, input_width, input_height)
            
            # Example: Overlay mask on frame
            overlaid_frame = overlay_mask_on_frame(frame, predictions)
            out.write(overlaid_frame)
            stframe.image(overlaid_frame[..., ::-1], channels="RGB", use_column_width=True)
        
        cap.release()
        out.release()
        
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f.read(), file_name="processed_video.mp4", mime="video/mp4")
