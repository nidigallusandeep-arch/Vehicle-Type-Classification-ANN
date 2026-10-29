# 🚘 Vehicle Type Classification Using ANN

## 📌 Project Overview

Vehicle Type Classification is a Deep Learning project that uses an Artificial Neural Network (ANN) to classify vehicle images into different categories.

Users can upload a vehicle image through a Streamlit web application and receive a predicted vehicle category along with its prediction confidence.

## 🎯 Project Objectives

- Classify vehicle images into multiple categories.
- Apply image preprocessing and normalization.
- Build and evaluate an Artificial Neural Network.
- Develop an interactive web application using Streamlit.
- Deploy the application for online access.

## 🗂️ Vehicle Categories

The dataset contains images from the following categories:

- Auto Rickshaws
- Bikes
- Cars
- Motorcycles
- Planes
- Ships
- Trains

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy
- **Image Processing:** Pillow (PIL)
- **Machine Learning Utility:** Scikit-learn
- **Web Application:** Streamlit
- **Development Environment:** Google Colab, VS Code
- **Version Control:** Git and GitHub

## ⚙️ Project Workflow

1. Collect vehicle image datasets.
2. Load images from category-specific folders.
3. Resize images to 64 × 64 pixels.
4. Convert images into numerical arrays.
5. Normalize pixel values between 0 and 1.
6. Encode vehicle category labels.
7. Split the dataset into training and testing sets.
8. Build and train the ANN model.
9. Evaluate the model using test accuracy.
10. Save the trained model and label encoder.
11. Build a Streamlit application for image prediction.

## 🧠 ANN Model Architecture

The model uses fully connected neural network layers to classify image data.

- Input: Resized RGB images
- Flatten Layer: Converts image data into a one-dimensional vector
- Dense Layers: Learn image feature patterns
- Batch Normalization: Helps stabilize training
- Dropout: Helps reduce overfitting
- Output Layer: Predicts probabilities for vehicle categories

## 📊 Model Evaluation

The model is evaluated using:

- Test Accuracy
- Training and Validation Accuracy
- Training and Validation Loss
- Predicted Vehicle Category
- Prediction Confidence

*Actual test accuracy should be added after evaluating the final trained model.*

## 📁 Project Structure

```text
Vehicle-Type-Classification-ANN/
│
├── app.py
├── requirements.txt
├── vehicle_ann_model.keras
├── vehicle_label_encoder.pkl
├── .gitignore
└── README.md
```

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/nidigallusandeep-arch/Vehicle-Type-Classification-ANN.git
```

### 2. Navigate to the project folder

```bash
cd Vehicle-Type-Classification-ANN
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment on macOS

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project files to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select the `main` branch.
5. Set the main file path to `app.py` if it is in the repository root.
6. Deploy the application.

Ensure that the model and label encoder files are accessible and that the deployment dependencies are compatible with the saved model.

## 🔮 Future Improvements

- Use Convolutional Neural Networks (CNNs) for better image feature extraction.
- Apply transfer learning using pretrained image classification models.
- Improve accuracy through data augmentation and hyperparameter tuning.
- Add a confusion matrix and classification report.
- Deploy the application with a public URL.

## 👨‍💻 Author

**Nidigallu Sandeep**

Computer Science and Machine Learning (CSM)

GitHub: [nidi gallusandeep-arch](https://github.com/nidigallusandeep-arch)

---

⭐ If you find this project useful, consider starring the repository.
