# Computer Vision Module - Campus Lectures

A comprehensive collection of Jupyter notebooks and projects demonstrating core computer vision and deep learning concepts. This module covers image classification, object detection, face recognition, and natural language processing applications built with TensorFlow, PyTorch, and OpenCV.

## Overview

This repository contains hands-on implementations of various computer vision and machine learning techniques, including:

- **Image Classification**: Deep learning models for image recognition
- **Face Recognition**: Attendance system using facial recognition
- **Object Detection**: YOLO-based detection pipelines
- **Data Augmentation**: Techniques for improving model robustness
- **Text Classification**: IMDB sentiment analysis using various approaches
- **Handwritten Character Recognition**: OCR-style digit/character recognition
- **Time Series Prediction**: Diabetes prediction from medical data

## Project Structure

```
computer-vision/
├── notebooks/
│   ├── deep_learning_image_classification.ipynb
│   ├── face_recognition_based_attendance.ipynb
│   ├── hand_written_character_recognition.ipynb
│   ├── data_augmentation_comparison.ipynb
│   ├── imdb_movie_review_classification_custom_keras.ipynb
│   ├── imdb_movie_review_classification_modern.ipynb
│   ├── imdb_movie_review_classification_sequential_tf_hub.ipynb
│   ├── imdb_moview_review_classification_tf_hub.ipynb
│   ├── diabetes_prediction.ipynb
│   ├── app.py
│   └── final_improvement_report.md
├── main.py
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd computer-vision
```

2. **Install dependencies**

Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -e .
```

3. **Activate virtual environment** (if using venv)
```bash
source .venv/bin/activate
```

## Notebooks Overview

### 1. **Deep Learning Image Classification**
`deep_learning_image_classification.ipynb`
- CNN architecture implementation
- Image dataset loading and preprocessing
- Model training and evaluation
- Transfer learning techniques

### 2. **Face Recognition-Based Attendance**
`face_recognition_based_attendance.ipynb`
- Face detection and recognition pipeline
- Attendance system implementation
- Real-time facial recognition
- Face embedding generation

### 3. **Handwritten Character Recognition**
`hand_written_character_recognition.ipynb`
- MNIST/digit recognition
- Neural network architecture
- Data preprocessing for OCR tasks
- Model evaluation and visualization

### 4. **Data Augmentation Comparison**
`data_augmentation_comparison.ipynb`
- Image augmentation techniques
- Effect of augmentation on model performance
- Different augmentation strategies
- Robustness improvements

### 5. **IMDB Movie Review Classification**
Multiple implementations of sentiment analysis:
- `imdb_movie_review_classification_custom_keras.ipynb` - Custom Keras implementation
- `imdb_movie_review_classification_modern.ipynb` - Modern approaches
- `imdb_movie_review_classification_sequential_tf_hub.ipynb` - TensorFlow Hub transfer learning
- `imdb_moview_review_classification_tf_hub.ipynb` - Alternative TF Hub approach

### 6. **Diabetes Prediction**
`diabetes_prediction.ipynb`
- Medical data analysis
- Time series and regression prediction
- Feature engineering
- Model evaluation metrics

## Key Dependencies

### Deep Learning Frameworks
- **TensorFlow** (2.21.0+) - Deep learning framework
- **PyTorch** (2.11.0+) - Machine learning framework
- **Keras** - High-level neural networks API (included with TensorFlow)

### Computer Vision
- **OpenCV** (4.13.0+) - Image processing
- **scikit-image** (0.26.0+) - Image processing algorithms
- **Pillow** (12.2.0+) - Image manipulation
- **ultralytics** (8.4.48+) - YOLO object detection

### Data & ML Tools
- **NumPy** (2.4.4+) - Numerical computing
- **Pandas** (3.0.2+) - Data manipulation
- **scikit-learn** (1.8.0+) - Machine learning utilities
- **TensorFlow Hub** (0.16.1+) - Pre-trained models
- **Transformers** (4.48.0+) - NLP models

### Utilities
- **Jupyter Lab** (4.5.7+) - Notebook environment
- **Matplotlib** (3.10.9+) - Plotting
- **Streamlit** (1.58.0+) - Web app framework
- **KaggleHub** - Dataset access

## Usage

### Running Jupyter Notebooks

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Navigate to the `notebooks/` directory and open any `.ipynb` file

3. Run cells sequentially or use "Run All" to execute the entire notebook

### Running the Application

If there's a Streamlit app in the notebooks:
```bash
streamlit run notebooks/app.py
```

Or run the main script:
```bash
python main.py
```

## Learning Outcomes

After working through these notebooks, you will understand:

- ✅ Fundamentals of deep learning and neural networks
- ✅ Convolutional Neural Networks (CNNs) for image tasks
- ✅ Transfer learning and pre-trained models
- ✅ Image preprocessing and data augmentation
- ✅ Real-world applications: face recognition, object detection
- ✅ Text classification and sentiment analysis
- ✅ Model evaluation and performance metrics
- ✅ Best practices for training and deploying ML models

## Environment Details

- **Python Version**: 3.12+
- **Development Environment**: Jupyter Lab
- **GPU Support**: CUDA-compatible (optional)

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)

## Notes

- Some notebooks may require downloading datasets from Kaggle (requires API credentials)
- GPU acceleration is recommended for large models and datasets
- See `final_improvement_report.md` for performance optimizations and improvements

## License

This repository contains educational materials for campus lectures.

---

**Last Updated**: June 2026
**Python**: 3.12+
**Status**: Active Development
