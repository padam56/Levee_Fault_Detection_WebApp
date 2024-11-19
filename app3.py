import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
from metrics import mcc_loss, mcc_metric, dice_coef, dice_loss, f1, tversky, tversky_loss, focal_tversky_loss, bce_dice_loss_new, jaccard, bce_dice_loss
from SandBoilNet import PCALayer, spatial_pooling_block, attention_block, initial_conv2d_bn, old_attention_block, conv2d_bn, iterLBlock, decoder_block, SandboilNet_Dropout

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

# Load models with custom objects
sandboil_model = load_model('sandboil_best_model.h5', custom_objects=custom_objects)
seepage_model = load_model('seepage_best_model.h5', custom_objects=custom_objects)

# UI Setup for Streamlit
st.title("Sandboil and Seepage Detection")

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
st.sidebar.write("### Legend")
if sandboil_selected:
    st.sidebar.markdown("<span style='color:green'>■ Sandboils</span>", unsafe_allow_html=True)
if seepage_selected:
    st.sidebar.markdown("<span style='color:pink'>■ Seepage</span>", unsafe_allow_html=True)
if crack_selected:
    st.sidebar.markdown("<span style='color:blue'>■ Crack</span>", unsafe_allow_html=True)
if potholes_selected:
    st.sidebar.markdown("<span style='color:orange'>■ Potholes</span>", unsafe_allow_html=True)

# Automatically adjust input size based on selected model
if sandboil_selected:
    input_width, input_height = 512, 512  # Dimensions for Sandboils model
elif seepage_selected:
    input_width, input_height = 256, 256  # Dimensions for Seepage model
    
    

# Additional customization options for image resolution and color variations in the sidebar
st.sidebar.write("### Customize Image Processing")

# Slider for adjusting image resolution scaling factor in the sidebar
resolution_factor = st.sidebar.slider("Adjust Image Resolution Scaling Factor", 0.1, 2.0, 1.0)

# Brightness and Contrast adjustment sliders
brightness_factor = st.sidebar.slider("Adjust Brightness", -100, 100, 0)
contrast_factor = st.sidebar.slider("Adjust Contrast", -100, 100, 0)

# Blurring option
blur_amount = st.sidebar.slider("Apply Gaussian Blur (Kernel Size)", 1, 15, 1, step=2)

# Edge detection option (Canny Edge Detection)
edge_detection = st.sidebar.checkbox("Apply Edge Detection (Canny)")

# Flip/Rotate options
flip_horizontal = st.sidebar.checkbox("Flip Horizontally")
flip_vertical = st.sidebar.checkbox("Flip Vertically")
rotate_angle = st.sidebar.slider("Rotate Image (Degrees)", -180, 180, 0)


def process_image(image):
    # Apply resolution scaling
    new_width = int(input_width * resolution_factor)
    new_height = int(input_height * resolution_factor)

    # Resize the image based on the scaling factor
    image = cv2.resize(image, (new_width, new_height))

    # Adjust brightness and contrast
    if brightness_factor != 0 or contrast_factor != 0:
        image = cv2.convertScaleAbs(image, alpha=1 + contrast_factor / 100.0, beta=brightness_factor)

    # Apply Gaussian blur if selected
    if blur_amount > 1:
        image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Apply edge detection if selected
    if edge_detection:
        image = cv2.Canny(image, threshold1=100, threshold2=200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Convert back to BGR so it can be displayed

    # Flip horizontally or vertically if selected
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)

    # Rotate the image by a specified angle
    if rotate_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    return np.expand_dims(image / 255.0 , axis=0) # Normalize and add batch dimension



#def process_image(image):
#    image = cv2.resize(image, (input_width, input_height))  # Resize the image
#    image = image / 255.0  # Normalize
#    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to apply the model and get predictions
from tensorflow.keras import backend as K

def apply_model(selected_model, image):
    processed_image = process_image(image)
    predictions = selected_model.predict(processed_image)
    
    # Assuming the output is a segmentation mask of shape (1, height, width, channels)
    predicted_mask = np.squeeze(predictions)  # Remove batch and channel dimensions if necessary
    K.clear_session()  # Clear session to free up memory
    return predicted_mask

# Function to overlay mask on image with color and intensity options
def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """ 
    Overlays a segmentation mask on an image with customizable transparency and color.
    :param image: Original image (numpy array)
    :param mask: Predicted mask (numpy array)
    :param alpha: Transparency factor for blending (0.0 - 1.0)
    :param color: Color of the overlay in BGR format (default is green)
    :return: Image with overlay
    """
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
    
    return overlaid_image

# Slider to control overlay intensity (transparency)
overlay_intensity = st.sidebar.slider("Overlay Intensity", 0.0, 1.0, 0.5)


# Detect sandboil functionality
if sandboil_selected:
    st.write("Selected Model: Sandboil Detection")
    
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png'])
    
    if uploaded_image is not None:
        # Read and decode uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Apply model to get predictions (segmentation mask)
        predictions = apply_model(sandboil_model, image)  # Assuming this returns a segmentation mask
        
        # Overlay the predicted mask on the original image with transparency control (alpha)
        overlaid_image = overlay_mask_on_image(image, predictions, alpha=overlay_intensity, color=(0, 255, 0))
        
        # Display original and overlaid images in Streamlit app interface
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(overlaid_image, caption='Image with Predicted Mask Overlay', use_column_width=True)

# Detect seepage functionality (similar logic for seepage detection)
if seepage_selected:
    st.write("Selected Model: Seepage Detection")
    
    uploaded_image = st.file_uploader("Upload an Image for Seepage Detection", type=['jpg', 'png'])
    
    if uploaded_image is not None:
        # Read and decode uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Apply model to get predictions (segmentation mask)
        predictions = apply_model(seepage_model, image)  # Assuming this returns a segmentation mask
        
        # Overlay the predicted mask on the original image with transparency control (alpha)
        overlaid_image = overlay_mask_on_image(image, predictions, alpha=overlay_intensity, color=(255, 105, 180))
        
        # Display original and overlaid images in Streamlit app interface
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(overlaid_image, caption='Image with Predicted Mask Overlay', use_column_width=True)



# Initialize confidence scores dictionary with all checkboxes set to zero initially
confidence_scores = {
    'Sandboil': 0,
    'Seepage': 0,
    'Crack': 0,
    'Potholes': 0,
    'Encroachment': 0,
    'Rutting': 0,
    'Animal Burrow': 0,
    'Vegetation': 0
}

        
uploaded_image = None
if uploaded_image is not None:
    # Load and display uploaded image
    image_data = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Process and display customized version of the uploaded image
    processed_image_data_displayed = process_image(image_data)[0] * 255.0 # Convert back from normalized form for display purposes only.
    processed_image_data_displayed = processed_image_data_displayed.astype(np.uint8) # Convert back to uint8 format.

    # Use tabs to separate content into different sections for better visibility
    tab1, tab2 = st.tabs(["Processed Image", "Confidence Scores"])

    with tab1:
        # Display the processed image in the first tab
        st.image(processed_image_data_displayed, caption='Processed Image', use_column_width=True)

    with tab2:
        # Placeholder for predictions (replace with actual model predictions)
        if sandboil_selected:
            confidence_scores['Sandboil'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if seepage_selected:
            confidence_scores['Seepage'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if crack_selected:
            confidence_scores['Crack'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if potholes_selected:
            confidence_scores['Potholes'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if encroachment_selected:
            confidence_scores['Encroachment'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if rutting_selected:
            confidence_scores['Rutting'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if animal_burrow_selected:
            confidence_scores['Animal Burrow'] = np.random.uniform(0.5, 1.0) * confidence_weight
        
        if vegetation_selected:
            confidence_scores['Vegetation'] = np.random.uniform(0.5, 1.0) * confidence_weight

        # Display results using a horizontal bar chart (show all checkboxes even if not selected)
        labels = list(confidence_scores.keys())
        scores = list(confidence_scores.values())

        fig, ax = plt.subplots()
        ax.barh(labels,scores)
        ax.set_xlabel('Confidence Score')
        ax.set_title('Detection Confidence Scores')

        # Display the bar chart in Streamlit after uploading an image in second tab
        st.pyplot(fig)

        # Display textual representation of detected classes and their respective confidences
        for label in labels:
            score = confidence_scores[label]
            if score > 0: # Only show detected items with non-zero scores.
                st.write(f"Detected: **{label}** with confidence: **{score:.2f}**")

## Enhanced Feedback Mechanism ##
def feedback_form():
    """ Function to create a feedback form with customizable options """
    
    feedback_option = st.radio(
        "Was this detection correct?",
        ("Yes", "No"), index=0)

    detailed_feedback = st.selectbox(
        "Please give details about the error:",
        ("None", "False Positive", "Missed Detection", "Ambiguous Result"), index=0)

    additional_comments=st.text_area(
        "Additional Comments (Optional):")

    submit_feedback=st.button('Submit Feedback')

    if submit_feedback:
        feedback_data={
            'Correct Detection':[feedback_option],
            'Detail':[detailed_feedback],
            'Comments':[additional_comments]
        }

        feedback_df=pd.DataFrame(feedback_data)
        
        feedback_file='feedback.csv'
        
        try:
            existing_feedback=pd.read_csv(feedback_file)
            updated_feedback=pd.concat([existing_feedback ,feedback_df], ignore_index=True)
            updated_feedback.to_csv(feedback_file ,index=False)
        
        except FileNotFoundError:
            feedback_df.to_csv(feedback_file ,index=False)

        st.success('Thank you for your feedback!')

feedback_form()



