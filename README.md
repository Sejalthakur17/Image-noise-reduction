# Noise Reduction App 🎯

A simple **Streamlit web application** for performing **image denoising / noise reduction** using popular OpenCV filters.  
Users can upload an image, apply different noise reduction algorithms, preview results, and download the processed image.

---

## 🚀 Features

- Upload images (`.jpg`, `.jpeg`, `.png`)
- Apply multiple noise reduction filters:
  - Gaussian Blur Filter
  - Median Blur Filter
  - Minimum Blur Filter (Erosion)
  - Maximum Blur Filter (Dilation)
  - Non-local Means Filter
- Adjustable kernel size
- Side-by-side comparison (Original vs Denoised)
- Download processed image in PNG / JPG / JPEG format

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – Web UI
- **OpenCV** – Image processing
- **NumPy** – Array operations
- **Pillow (PIL)** – Image handling

---

Base code from PythonGUIs tutorial; Docker and CI/CD setup by me.

## 📦 Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .github/workflow
├── streamlit.out
├── dev-status

