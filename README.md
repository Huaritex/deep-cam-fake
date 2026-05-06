# Deep Live Cam (Modernized) 🚀

Real-time face swap and high-fidelity video deepfakes with a single click. This version features a **modernized sidebar-based UI**, optimized performance for NVIDIA RTX 40-series, and a specialized **Low-Light Mode** for challenging environments.

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## 🌟 Modernized Edition Highlights

This project has been updated by **@huaritex-software** with a focus on professional aesthetics and extreme performance:

-   **Professional UI/UX:** A brand new, minimalist sidebar-based interface for a cleaner workflow.
-   **Low-Light Mastery:** Adaptive CLAHE algorithm to handle dark environments (perfect for university fairs or low-light rooms).
-   **Hardware Optimized:** Specifically tuned for **NVIDIA RTX 4060** and **Intel Core i7 13th Gen** using CUDA-accelerated processing.
-   **Multi-Threaded Architecture:** Decoupled capture, detection, and processing threads to achieve up to **60 FPS** in live mode.

---

## 📸 University Fair Showcase (Special Context)

This project is optimized for university technology fairs. The **Low-Light Mode** allows the stand to be set up in "dark room" or "lounge" environments without losing face tracking quality.

1.  **QR Integration:** Students can upload their photos via a QR code, which the operator then selects as the "Source Face".
2.  **Live Interaction:** Real-time transformation on a monitor, showing the power of Software Engineering and AI.

---

## 🛠 Features

-   **Mouth Masking:** Retain original mouth movements for realistic speech.
-   **Face Mapping:** Swap multiple subjects simultaneously with different source faces.
-   **Face Enhancement:** Integrated support for **GFPGAN**, **GPEN-256**, and **GPEN-512** for high-definition results.
-   **Post-Processing:** Sharpness and transparency controls to blend results perfectly.
-   **Live Mirror:** Natural mirror-view for real-time interaction.

---

## 🔧 Installation (Optimized for RTX 4060)

### 1. Prerequisites
-   Python 3.11 (Recommended)
-   Git
-   [FFmpeg](https://ffmpeg.org/download.html) (Added to system PATH)
-   [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 2. Clone and Setup
```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. GPU Acceleration (NVIDIA)
For a setup with **RTX 4060**, ensure you have:
1.  [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
2.  [cuDNN v8.9.x](https://developer.nvidia.com/cudnn)

Run the following to ensure the correct ONNX runtime:
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.21.0
```

---

## 🚀 Usage

### Starting the Application
Simply run:
```bash
python run.py
```
*The app will automatically detect your NVIDIA GPU and use CUDA for all operations.*

### Quick Start Guide
1.  **Face Swap Tab:** Select your "Source Face" (your target) and the "Target Media" (your video or image).
2.  **Enhancement Tab:** Choose a model (e.g., GPEN-512) and adjust sharpness/transparency. 
    *   *Tip: Enable **Low Light Mode** if the room is dark.*
3.  **Live Mode Tab:** Select your camera and hit **START LIVE SESSION**.

---

## 📂 Models
The following models should be placed in the `/models` directory:
-   `inswapper_128_fp16.onnx` (Core Face Swapper)
-   `GFPGANv1.4.onnx` (Enhancer)
-   `gpen-256.onnx` / `gpen-512.onnx` (High-res Enhancers)

[Download Models from HuggingFace](https://huggingface.co/hacksider/deep-live-cam/tree/main)

---

## ⚖️ Disclaimer

This software is for educational and productive AI-generated media purposes. 
-   **Consent:** Obtain permission before using someone else's face.
-   **Ethics:** A built-in NSFW filter prevents the processing of inappropriate content.
-   **Responsibility:** The authors are not responsible for misuse of this tool.

---
