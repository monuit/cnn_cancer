# CNN Cancer Detection Project

## Overview

This project implements a Convolutional Neural Network (CNN) for the detection of metastatic tissue in histopathologic scans of lymph node sections. The project was developed as part of Week 3 coursework and uses deep learning techniques to classify medical images as either malignant or normal.

## Problem Description

The primary objective is to create an algorithm capable of identifying metastatic tissue in histopathologic scans with high accuracy. Specifically, the model needs to:

- Classify **57,458 test images** as either **malignant** or **normal**
- Detect if an image contains at least one pixel of tumor tissue in the center 32×32px region
- Achieve high classification accuracy for medical diagnostic purposes

## Dataset

The project utilizes the **PatchCamelyon (PCam)** benchmark dataset provided through Kaggle's Histopathologic Cancer Detection competition:

- **Training Dataset**: 220,025 images with known labels (malignant/normal)
- **Test Dataset**: 57,458 images for evaluation
- **Image Format**: 96×96×3 (RGB images)
- **Dataset Balance**: ~40% malignant cases, ~60% normal cases
- **Data Quality**: No duplicates (unlike original PCam dataset)

### Key Dataset Features
- ✅ Balanced dataset distribution
- ✅ Binary classification labels (0=normal, 1=malignant)
- ✅ No duplicate images in training or test sets
- ✅ Consistent image dimensions (96×96×3)

## Project Structure

```
week3_cnn_cancer_detection/
├── week3_cnn_cancer_detection.ipynb    # Main Jupyter notebook
├── .gitignore                          # Git ignore file
├── README.md                           # This file
├── history/                            # Training history (ignored)
├── models/                             # Saved models (ignored)
├── test/                               # Test data (ignored)
├── train/                              # Training data (ignored)
└── venv/                               # Virtual environment (ignored)
```

## Technical Implementation

### Architecture
The project implements CNN models using **TensorFlow/Keras** with the following components:

- **Input Layer**: 96×96×3 RGB images
- **Convolutional Layers**: Feature extraction with Conv2D layers
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Dense Layers**: Fully connected layers for classification
- **Dropout**: Regularization to prevent overfitting
- **Output Layer**: Binary classification (sigmoid activation)

### Key Libraries and Dependencies

```python
- tensorflow/keras          # Deep learning framework
- pandas                   # Data manipulation
- numpy                    # Numerical computations
- matplotlib               # Data visualization
- PIL                      # Image processing
- scikit-learn            # Machine learning utilities
- IPython.display         # Jupyter notebook utilities
```

### Data Processing Pipeline

1. **Data Loading**: Efficient loading of large image datasets
2. **Data Augmentation**: Using ImageDataGenerator for improved generalization
3. **Preprocessing**: Normalization and resizing
4. **Train/Validation Split**: Proper data splitting for model evaluation
5. **Batch Processing**: Optimized batch processing for training efficiency

### Model Training Features

- **Early Stopping**: Prevents overfitting with patience monitoring
- **Model Checkpointing**: Saves best performing models
- **Training History**: Comprehensive logging of training metrics
- **Validation Monitoring**: Real-time validation accuracy tracking

## Exploratory Data Analysis (EDA)

The notebook includes comprehensive EDA covering:

- **Class Distribution Analysis**: Examination of malignant vs. normal case ratios
- **Image Visualization**: Sample image display from both classes
- **Data Quality Assessment**: Verification of no duplicates and consistent formatting
- **Statistical Summary**: Dataset size and dimension analysis

## Model Performance

The project includes multiple model architectures and training approaches:

- **Basic CNN Model**: Baseline implementation
- **Enhanced Models**: Improved architectures with regularization
- **Performance Comparison**: Analysis of different model configurations
- **Visualization**: Training history plots showing accuracy and loss trends

## Getting Started

### Prerequisites

```bash
Python 3.7+
TensorFlow 2.x
Jupyter Notebook
Required Python packages (see imports in notebook)
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd week3_cnn_cancer_detection
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install tensorflow pandas numpy matplotlib pillow scikit-learn jupyter
   ```

4. **Download dataset**: 
   - Access the Kaggle competition data
   - Place training images in `train/` directory
   - Place test images in `test/` directory

### Running the Project

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook week3_cnn_cancer_detection.ipynb
   ```

2. **Execute cells sequentially**: 
   - Follow the notebook structure from EDA through model training
   - Adjust hyperparameters as needed
   - Monitor training progress and validation metrics

## File Structure Details

### Data Organization
- `train/`: Training images organized by class
- `test/`: Test images for final evaluation  
- `models/`: Saved trained models
- `history/`: Training history and logs

### Excluded Files (.gitignore)
- Large datasets (`*.csv`, `*.zip`)
- Model files and training artifacts
- Virtual environment
- System-specific files

## Key Features

- 🔬 **Medical Image Classification**: Specialized for histopathologic analysis
- 🧠 **Deep Learning**: State-of-the-art CNN architectures
- 📊 **Comprehensive Analysis**: From EDA to model evaluation
- 🎯 **High Accuracy**: Optimized for medical diagnostic accuracy
- 📈 **Visualization**: Rich plotting of training metrics and results
- ⚡ **Efficient Processing**: Optimized for large-scale image datasets

## Competition Context

This project was developed for the **Kaggle Histopathologic Cancer Detection** competition, focusing on:

- Binary classification of medical images
- High-stakes medical diagnostic applications
- Large-scale dataset processing
- Performance optimization for accuracy

## Results and Evaluation

The notebook includes detailed analysis of:
- Model accuracy metrics
- Training and validation loss curves
- Confusion matrices and classification reports
- Comparative analysis of different model architectures

## Future Improvements

Potential enhancements could include:
- Advanced data augmentation techniques
- Transfer learning with pre-trained models
- Ensemble methods for improved accuracy
- Cross-validation for robust evaluation
- Model interpretability analysis

## License

This project is developed for educational purposes as part of coursework assignments.

## References

- [Kaggle Histopathologic Cancer Detection Competition](https://www.kaggle.com/c/histopathologic-cancer-detection/overview)
- PatchCamelyon (PCam) benchmark dataset
- TensorFlow/Keras documentation

---

**Note**: This project demonstrates practical application of deep learning techniques to medical image analysis, emphasizing both technical implementation and real-world diagnostic accuracy.