# Live Object Detection using YOLO v11

## 📌 Overview
This project implements Real-time Object Detection using YOLO v11. The system can detect multiple objects from a live camera feed or video stream with high accuracy and speed, making it suitable for applications like surveillance, robotics, and safety systems.

The model is built using PyTorch, OpenCV, and the Ultralytics YOLO library. It supports CUDA for hardware acceleration to enable real-time performance.

---

## 📂 Table of Contents
- 🚀 [Installation](#installation)
- ⚙️ [Requirements](#requirements)
- 📸 [Usage](#usage)
- 🔍 [How it Works](#how-it-works)
- 📁 [File Structure](#file-structure)
- 🎯 [Model Weights](#model-weights)
- 🐞 [Troubleshooting](#troubleshooting)
- 🤝 [Contributing](#contributing)
- 📜 [License](#license)

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Ayushskull7/LiveObjectDetection-YOLO-v11.git
cd live-object-detection-yolo-v11
```

### 2️⃣ Set up Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Download YOLO v11 Weights
Modify the model's name in `live_object_detection.py` to automatically download pretrained YOLO v11 weights.

---

## ⚙️ Requirements
- Python 3.7+
- PyTorch 1.8+ with CUDA support
- OpenCV 4.5+
- Ultralytics YOLO

---

## 📸 Usage

### ▶️ Run Live Object Detection
```sh
python scripts/live_object_detection.py --source 0
```
- `--source 0`: Uses the default webcam. You can change it to a video file or RTSP stream.

### 💾 Save Output Video
```sh
python scripts/live_object_detection.py --source 0 --save output.mp4
```

### 🎯 Detect Specific Objects
Modify `objects.txt` with target object names, then run the script.

---

## 🔍 How it Works
1. Captures frames from the video source.
2. Preprocesses frames for YOLO inference.
3. YOLO detects objects and assigns bounding boxes.
4. Applies Non-Maximum Suppression (NMS) to refine results.
5. Displays objects with labels and confidence scores.

---

## 📁 File Structure
```
live-object-detection-yolo-v11/
│
├── models/                    # YOLO weights and configs
├── output/                    # Saved video outputs
├── scripts/
│   ├── live_object_detection.py    # Live detection script
│   ├── checkcuda.py                # Check if CUDA is supported
│   └── objects.txt                 # List of objects to detect
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
```

---

## 🎯 Model Weights
The project uses YOLO v11 for object detection. Pre-trained weights are downloaded automatically.

**Model Used:** `yolo11x.pt`

---

## 🐞 Troubleshooting

### 🔹 CUDA Errors:
Check GPU support and drivers using:
```sh
python -c "import torch; print(torch.cuda.is_available())"
```

### 🔹 Slow Processing:
Ensure GPU acceleration is enabled.

### 🔹 Camera Issues:
Verify webcam access and permissions.

---

## 🤝 Contributing
1. Fork the repository.
2. Create a new branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Make changes and commit:
   ```sh
   git commit -am 'New feature'
   ```
4. Push to your fork:
   ```sh
   git push origin feature-branch
   ```
5. Open a pull request.

---

## 📢 Acknowledgments
- **YOLOv11**: Based on the work by Joseph Redmon and the YOLO community.
- **Ultralytics YOLO**: For model implementation and support.

🔗 For more details, check the [YOLO official documentation](https://github.com/ultralytics/ultralytics). 🚀
