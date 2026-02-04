# NeuroScan Pro - Brain Tumor Detection System

A modern desktop application built with Python, CustomTkinter, and YOLOv8 for automated brain tumor detection from MRI scans.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## 🌟 Features

- **Advanced Detection**: Utilizes a trained YOLOv8 model (`best.pt`) for high-accuracy segmentation.
- **Modern UI**: Built with CustomTkinter for a sleek, dark-themed interface.
- **Real-time Visualization**: Side-by-side comparison of original MRI and detection results.
- **User Friendly**: Simple upload and analyze workflow.

## 🛠️ Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/AbdulAhad5138/Brain_Tumer_Detection_system.git
   cd Brain_Tumer_Detection_system
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the main desktop application:

```bash
python desktop_app.py
```

## 📁 Project Structure

- `desktop_app.py`: Main application entry point (GUI).
- `best.pt`: Trained YOLOv8 model weights.
- `requirements.txt`: List of Python dependencies.
- `app.py`: Alternative/Legacy entry point (if applicable).

## 📝 Requirements

- customtkinter
- ultralytics
- pillow
- numpy
- opencv-python-headless
- packaging

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📜 License

[MIT](LICENSE)
