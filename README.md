#  Multimodal Crop Disease Detection Using Deep Learning

##  Project Overview

Crop diseases are a major challenge in agriculture, affecting crop quality, productivity and food security. Early identification of plant diseases can help farmers take preventive measures and reduce crop losses.

This project explores a **multimodal deep learning approach for crop disease classification**, combining visual information from plant leaf images with supplementary weather information.

The primary objective is to investigate whether integrating image-based features with environmental data can provide a more informative approach to crop disease classification compared to using image data alone.

The project uses a pretrained ResNet50 model for image-based disease classification, an LSTM-based network for weather feature processing, and different multimodal fusion strategies to explore the combination of these two modalities.

##  Project Objectives

- Develop a deep learning model to classify plant diseases from leaf images.
- Apply transfer learning using a pretrained ResNet50 architecture.
- Explore weather data as an additional source of environmental information.
- Investigate early fusion and weighted late fusion techniques.
- Compare image-only classification with multimodal model experiments.
- Apply Grad-CAM to understand the image regions influencing model predictions.

##  Key Idea

Traditional image-based crop disease detection systems primarily depend on visual symptoms present on plant leaves.

However, environmental conditions such as temperature, humidity and precipitation can also influence disease development.

This project explores the idea of combining:

1. **Visual Modality:** Plant leaf images containing disease-related visual patterns.
2. **Environmental Modality:** Weather observations containing relevant environmental parameters.

By extracting features from both modalities and exploring different fusion strategies, the project investigates the potential of multimodal learning for crop disease classification.

##  System Architecture

The experimental workflow consists of the following stages:

### 1. Image Data Processing

- Load plant leaf images using PyTorch ImageFolder.
- Resize images to 224 × 224 pixels.
- Convert images into tensors.
- Divide the dataset into training, validation and testing sets.

### 2. Image-Based Disease Classification

A pretrained **ResNet50** architecture is used as the baseline image classification model.

- Transfer learning using ImageNet-pretrained weights.
- Replace the original classification layer with a custom fully connected layer.
- Freeze the pretrained feature extraction layers during baseline training.
- Train the final classification layer to identify crop disease categories.

### 3. Weather Data Processing

Weather observations obtained from NASA POWER are used as supplementary environmental information.

The preprocessing pipeline includes:

- Loading weather observations from CSV files.
- Removing unnecessary date-related columns.
- Handling missing values.
- Converting numerical weather features into PyTorch tensors.

An LSTM-based neural network is explored for processing the weather features.

### 4. Multimodal Feature Fusion

Two fusion strategies are investigated.

**A. Early Fusion**

Early fusion combines the feature representations extracted from the image and weather branches before classification.

- ResNet50 extracts a 2048-dimensional image feature vector.
- An LSTM-based network produces a 32-dimensional weather feature vector.
- Both feature vectors are concatenated.
- The combined representation is passed through fully connected layers for classification.

**B. Weighted Late Fusion**

Late fusion combines the classification outputs from the image and weather models.

- Obtain class probabilities from the image model.
- Obtain class probabilities from the weather model.
- Combine both predictions using a weighted average.

The fusion equation is:

`P_final = α × P_image + (1 − α) × P_weather`

Where:

- `P_final` = Combined prediction probabilities.
- `P_image` = Image model probabilities.
- `P_weather` = Weather model probabilities.
- `α` = Weight assigned to the image modality.

This allows the relative contribution of each modality to be controlled.

### 5. Model Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is explored to visualize the image regions contributing to the model's predictions.

It helps provide visual insights into which parts of a plant leaf influence the classification decision.

### 6. Model Evaluation

The notebook explores the following evaluation methods:

- Classification accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Class-wise performance analysis
- Comparison between baseline and fusion experiments

##  Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision, OpenCV |
| Image Classification | ResNet50 |
| Sequence Modeling | LSTM |
| Multimodal Learning | Early Fusion, Weighted Late Fusion |
| Explainable AI | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Weather Data Source | NASA POWER |
| Development Environment | Jupyter Notebook / Google Colab |

##  Dataset Information

### Plant Leaf Image Dataset

The project uses a folder-based plant leaf disease image dataset containing **39 disease and healthy plant categories**.

The images are organized into class-specific directories and loaded using PyTorch's ImageFolder utility.

### Weather Dataset

Weather observations are obtained from NASA POWER.

The weather dataset provides environmental measurements that are explored as supplementary input features.

**Note:** The image dataset and weather CSV are not included in this repository. Refer to the dataset instructions for local setup.

##  Project Structure

```text
Multimodal-Crop-Disease-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── DATASET.md
│
└── notebooks/
    └── final_codebook.ipynb
```

##  Installation and Setup

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook or Google Colab
- CUDA-compatible GPU (recommended for faster training)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Multimodal-Crop-Disease-Detection.git

cd Multimodal-Crop-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU-based execution, install the appropriate PyTorch and CUDA versions according to your system configuration.

### 4. Configure the Dataset

- Download and arrange the plant leaf dataset into class-specific folders.
- Place the weather CSV in the appropriate local directory.
- Update the dataset and weather file paths in the notebook.

Refer to `DATASET.md` for the expected folder structure.

### 5. Run the Notebook

```bash
jupyter notebook notebooks/final_codebook.ipynb
```

Execute the notebook cells sequentially to explore the image classification, weather processing, fusion experiments and explainability pipeline.

##  Experimental Scope and Limitations

This repository contains an experimental implementation exploring multimodal crop disease classification.

The current notebook has certain limitations that should be considered when interpreting the results:

- Weather labels are randomly generated for the weather-only classification experiment and are not actual disease labels.
- Weather observations are not aligned with individual leaf images using location and capture-date metadata.
- A mean weather feature vector is repeated across image batches in the fusion experiments.
- The weather LSTM currently receives a sequence of length one, so temporal dependencies are not meaningfully modeled.
- The fusion experiments require further validation with properly aligned multimodal data and independently trained branches.

Therefore, the current implementation demonstrates an exploration of multimodal architectures and fusion strategies rather than a validated improvement in crop disease prediction through weather integration.

##  Future Enhancements

- Establish sample-level alignment between plant images and weather observations.
- Incorporate actual environmental measurements associated with image collection locations and dates.
- Develop a properly trained temporal weather prediction branch.
- Experiment with attention-based multimodal fusion.
- Explore transformer-based architectures for improved feature representation.
- Evaluate the models using leakage-safe data splits and independent test datasets.
- Improve model interpretability and visualization.
- Develop a real-time crop disease prediction interface.

##  Author

**Nikhita Ajith**
---

*This project was developed as part of an academic exploration of deep learning, computer vision and multimodal learning for agricultural applications.*
