# Hand-Gesture-CV-r ✋📹

> Real-time Computer Vision hand segmentation and finger counting pipeline built with OpenCV, background subtraction, convex hull geometry, and contour analysis.

## 📌 Overview
**Hand-Gesture-CV-r** is an edge-optimized classical Computer Vision system that detects human hands and accurately counts raised fingers from live webcam video feeds in real time without requiring heavy GPU-dependent deep neural networks.

## 🧠 Core Computer Vision Concepts
1. **Background Accumulation Modeling:** Running weighted average accumulation (`cv2.accumulateWeighted`) dynamically adapts to subtle lighting shifts.
2. **Foreground Segmentation:** Absolute image difference (`cv2.absdiff`) + binary thresholding to isolate hand motion contours from static environments.
3. **Convex Hull & Extreme Points:** Calculating polygon envelopes enclosing hand contours to identify geometric endpoints (tips of fingers, wrist boundaries).
4. **Distance Transform & Circular Masking:** Establishing centroid reference centers and creating geometric intersection circles to evaluate defect angles and finger count.

## 📁 Repository Structure
```text
├── finger_counter.py      # Standalone CLI live webcam stream runner
├── Finger_Count.ipynb     # Step-by-step interactive Jupyter notebook
├── hand_convex.png        # Mathematical geometry visualization reference
├── requirements.txt       # Dependencies
└── README.md
```

## ⚙️ Setup & Execution

### 1. Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Live Webcam Stream
```bash
python finger_counter.py
```
*Note: Keep your hand outside the green bounding box for the first 60 frames during background calibration. Once calibration finishes, place your hand inside the ROI.* Press `Esc` to quit.

### 3. Interactive Notebook
```bash
jupyter lab Finger_Count.ipynb
```

## 📄 License
MIT License
