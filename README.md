# 🧠 Brain Tumor Detection using Deep Learning (VGG16)

A deep learning project that detects and classifies brain tumors from MRI images using Transfer Learning with the VGG16 Convolutional Neural Network.

---

## 📌 Overview

This project uses a pretrained VGG16 model to classify brain MRI images into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The model leverages transfer learning to achieve high accuracy with limited medical imaging data.

---

## 🚀 Features

- Uses pretrained VGG16 (Transfer Learning)
- Image preprocessing and normalization
- Image augmentation (brightness and contrast)
- Fine-tuning of pretrained layers
- Multi-class classification
- Model evaluation using confusion matrix and classification report
- Accurate tumor prediction from MRI scans

---

## 🧠 Model Architecture

```

Input Image (128x128x3)
│
▼
Pretrained VGG16 (Feature Extraction)
│
▼
Flatten Layer
│
▼
Dense Layer
│
▼
Dropout Layer
│
▼
Output Layer (Softmax - 4 classes)

```

---

## 📂 Dataset Structure

```

MRI Images/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/

````

---

## 🔄 Workflow

1. Load MRI image dataset
2. Preprocess images (resize and normalize)
3. Apply image augmentation
4. Load pretrained VGG16 model
5. Freeze base layers
6. Add custom classification layers
7. Train the model
8. Evaluate performance
9. Predict tumor type

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Scikit-learn

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
````

Install dependencies:

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook Brain_Tumor_Detection.ipynb
```

Or run Python script:

```bash
python brain_tumor_detection.py
```

---

## 📊 Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example output:

```
Accuracy: 95%
Precision: High
Recall: High
F1 Score: High
```

---

## 🧪 Sample Prediction Output

Input: MRI Image
Output:

```
Prediction: Pituitary Tumor
Confidence: 97%
```

---

## 📈 Transfer Learning

This project uses VGG16 pretrained on ImageNet.

Benefits:

* Faster training
* Higher accuracy
* Less training data required

---

## 🧠 Real-World Applications

* Medical diagnosis assistance
* Automated tumor detection systems
* Radiology AI tools
* Healthcare AI systems

---

## 🔮 Future Improvements

* Deploy as web app (Flask / Streamlit)
* Use EfficientNet / ResNet
* Improve dataset size
* Add Grad-CAM visualization
* Deploy on cloud

---

## 📷 Workflow Diagram

![Workflow](<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/3ed8f7eb-6bda-4127-8d89-b079e6b32b7b" />)

---

## 👨‍💻 Author

Vinayak Tiwari

