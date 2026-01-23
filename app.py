import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def image_to_array(file_to_open):
    """Load a Streamlit image into an array."""
    # Convert the uploaded file to a PIL image.
    image = Image.open(file_to_open)

    # Convert the image to an RGB NumPy array for processing.
    image = image.convert("RGB")
    image = np.array(image)
    return image


def denoise_image(image, algorithm, kernel_size):
    """Apply a denoising algorithm safely."""

    # ---------- IMAGE SAFETY ----------
    # Convert float images to uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert RGBA → RGB
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # ---------- KERNEL SAFETY ----------
    if kernel_size <= 0:
        kernel_size = 1

    if kernel_size % 2 == 0:
        kernel_size += 1

    # ---------- FILTERS ----------
    # --- Gaussian Blur ---
    if algorithm == "Gaussian Blur Filter":
        denoised_image = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), 0
        )

    # --- Median Blur ---
    elif algorithm == "Median Blur Filter":
        # Median blur requires kernel >= 3
        if kernel_size < 3:
            kernel_size = 3

        denoised_image = cv2.medianBlur(image, kernel_size)

    # --- Minimum Filter (Erosion) ---
    elif algorithm == "Minimum Blur Filter":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        denoised_image = cv2.erode(image, kernel, iterations=1)

    # --- Maximum Filter (Dilation) ---
    elif algorithm == "Maximum Blur Filter":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        denoised_image = cv2.dilate(image, kernel, iterations=1)

    # --- Non-local Means ---
    elif algorithm == "Non-local Means Filter":
        h = kernel_size * 5
        denoised_image = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, 7, 21
        )

    else:
        denoised_image = image

    return denoised_image




def image_array_to_bytes(image_to_convert, file_format):
    """Given an image array, convert it to a bytes object."""

    # Converting NumPy array to PIL image in RGB mode
    image_pil = Image.fromarray(image_to_convert)

    # Creating a buffer to store the image data in the selected format
    buf = io.BytesIO()
    if file_format.upper() == "JPG":
        file_format = "JPEG"
    image_pil.save(buf, format=file_format)
    byte_data = buf.getvalue()
    return byte_data


st.title("Noise Reduction App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

algorithm = st.selectbox(
    "Select noise reduction algorithm",
    (
        "Gaussian Blur Filter",
        "Median Blur Filter",
        "Minimum Blur Filter",
        "Maximum Blur Filter",
        "Non-local Means Filter",
    ),
)

kernel_size = st.slider("Select kernel size", 1, 10, step=2)


if uploaded_file is not None:
    image = image_to_array(uploaded_file)
    denoised_image = denoise_image(image, algorithm, kernel_size)

    # Displaying the denoised image in RGB format
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.image(denoised_image, caption="Denoised Image", use_container_width=True)

    # Dropdown to select the file format for downloading
    file_format = st.selectbox("Select output format", ("PNG", "JPEG", "JPG"))

    byte_data = image_array_to_bytes(denoised_image, file_format)

    # Button to download the processed image
    st.download_button(
        label="Download Image",
        data=byte_data,
        file_name=f"denoised_image.{file_format.lower()}",
        mime=f"image/{file_format.lower()}",
    )
