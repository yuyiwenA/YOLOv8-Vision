# YOLOv8-Vision
Video object detection pipeline built with YOLOv8 and OpenCV, supporting frame-wise inference and object trajectory export.

## ✨ Features
- End-to-end video object detection based on YOLOv8n
- Visualize bounding boxes and confidence scores on output video
- Export detection information (class, confidence, box coordinates) to CSV file
- Flexible local video input for custom scene test

## 🛠 Environment
- Python >=3.8
- ultralytics
- opencv-python
- pandas
- numpy

## 📁 File Structure
├── main.py                # core detection program
├── requirements.txt       # dependency list
├── test_video.mp4         # input test video
├── result.mp4             # output video with visualized boxes
└── trajectory.csv         # frame-level detection trajectory records

## 🚀 Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Put your test video in the folder and rename it to `test_video.mp4`, or modify the video path in main.py
3. Run the detection pipeline
```
python main.py
```

## 📌 Customized improvements
Based on the original YOLOv8 inference API:
- Integrated OpenCV video read-write pipeline
- Added frame-by-frame detection trajectory storage to CSV
- Added frame progress printing for long video processing

## 📝 Notes
- The YOLOv8n pre-trained weight will auto-download at the first run
- Supports mp4, avi, mov and other mainstream video formats
