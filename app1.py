import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# UI Setup for Streamlit
st.title("Levee Problems Detection WebApp")

# Sidebar for model selection and customization
st.sidebar.title("Options")

# Add a 'Select All' checkbox with custom styling and position it higher
st.markdown(
    """
    <style>
    .big-checkbox .stCheckbox {font-size: 24px !important;}
    </style>
    """, unsafe_allow_html=True
)

# Create a larger 'Select All' checkbox at the top
select_all = st.checkbox("Select All", key="select_all")

# Individual detection checkboxes, controlled by 'Select All'
sandboil_selected = st.checkbox("Detect Sandboils", value=select_all)
seepage_selected = st.checkbox("Detect Seepage", value=select_all)
crack_selected = st.checkbox("Detect Crack", value=select_all)
potholes_selected = st.checkbox("Detect Potholes", value=select_all)
encroachment_selected = st.checkbox("Detect Encroachment", value=select_all)
rutting_selected = st.checkbox("Detect Rutting", value=select_all)
animal_burrow_selected = st.checkbox("Detect Animal Burrow", value=select_all)
vegetation_selected = st.checkbox("Detect Vegetation", value=select_all)


# If any detection is selected, show the common confidence score slider in the sidebar
if any([sandboil_selected, seepage_selected, crack_selected, potholes_selected, encroachment_selected, rutting_selected, animal_burrow_selected, vegetation_selected]):
    confidence_weight = st.sidebar.slider("Adjust Detection Confidence Weight", 0.0, 1.0, 1.0)

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

# Function to process the image based on user customizations
input_width, input_height = 256, 256 # Adjust based on your model's input size

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

# Placeholder for model application (commented out)
def apply_model(selected_model, image):
    processed_image = process_image(image)
    predictions = selected_model.predict(processed_image)
    return predictions

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

if sandboil_selected or seepage_selected or crack_selected or potholes_selected or encroachment_selected or rutting_selected or animal_burrow_selected or vegetation_selected:
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png'])

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

