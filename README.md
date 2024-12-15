# **Retinal OCT Image Classification using DenseNet**

---

## **Project Overview**

This project focuses on **Retinal OCT (Optical Coherence Tomography) Image Classification** using the **DenseNet** architecture. OCT imaging is an essential tool for diagnosing various retinal diseases such as *Normal*, *Drusen*, *Cataract*, and *Diabetic Macular Edema (DME)*. By leveraging **DenseNet** and transfer learning, this project aims to accurately classify OCT images into relevant categories.

---

## **Project Objectives**

1. Build and train a **DenseNet-based Convolutional Neural Network** for OCT image classification.
2. Improve model performance using **transfer learning** and **data augmentation**.
3. Evaluate the model's performance using metrics such as **accuracy, precision, recall, and F1 score**.
4. Visualize model predictions and explain results using **Grad-CAM**.
5. Provide a deployable model for real-world OCT classification tasks.

---

## **Technologies and Tools Used**

- **Python**: Programming Language
- **TensorFlow/Keras**: Model building, training, and evaluation
- **NumPy, Pandas**: Data manipulation and preprocessing
- **Matplotlib, Seaborn**: Visualization of results and model performance
- **OpenCV**: Image processing
- **Grad-CAM**: Model explainability visualization
- **Flask**: Backend deployment for web interface
- **Jupyter Notebook**: Development environment for iterative coding
- **HDF5**: Model saving format
- **Postman**: API testing for deployment

---

## **Dataset**

The **Retinal OCT dataset** used in this project contains images categorized into the following classes:
1. **Normal**  
2. **Drusen**  
3. **Cataract**  
4. **DME**  

### **Preprocessing**:
- Image resizing to a consistent resolution (e.g., 224x224).
- Normalization of pixel values.
- Splitting into **train**, **validation**, and **test** sets.

---

## **Model Architecture**

The project uses **DenseNet** (Dense Convolutional Network), a deep learning model that connects each layer to every other layer to improve feature reuse and reduce overfitting.

- **Key Features**:
  - Pre-trained DenseNet model (e.g., DenseNet121 on ImageNet) for transfer learning.
  - Custom classification head to match the number of classes (softmax activation).
  - Optimization using **Adam Optimizer**.
  - Use of **EarlyStopping** and **ModelCheckpoint** callbacks for better training control.

---

## **Project Workflow**

### 1. **Data Preparation**
- Image resizing, normalization, and splitting into train/validation/test sets.
- Data augmentation for improving generalization.

### 2. **Model Building**
- Loaded pre-trained DenseNet model.
- Added custom layers for classification.
- Configured the model for training.

### 3. **Training**
- Trained the model using augmented data.
- Monitored accuracy and loss with callbacks.

### 4. **Evaluation**
- Evaluated on the test set with metrics like accuracy, precision, recall, and F1 score.
- Generated confusion matrix and ROC curve for deeper analysis.

### 5. **Model Explainability**
- Applied **Grad-CAM** to visualize the regions in the OCT images where the model focused during predictions.

### 6. **Deployment**
- Deployed the trained model using Flask for real-time image classification.

---

## **Performance Metrics**

- **Accuracy**: Achieved high accuracy on test data.  
- **F1-Score**: Ensured balanced performance across all classes.  
- **Confusion Matrix**: Visualized the performance of the model per class.  
- **ROC Curve**: Analyzed the trade-off between true positives and false positives.  

---

## **Challenges Faced**

1. **Dataset Imbalance**: Classes with fewer samples led to biased predictions.
   - *Solution*: Used weighted loss functions and data augmentation.

2. **High Computational Load**: Training DenseNet on OCT images is resource-intensive.
   - *Solution*: Utilized GPU acceleration.

3. **Interpretability**: Understanding predictions was challenging.
   - *Solution*: Implemented Grad-CAM to explain model decisions.

4. **Deployment Integration**: Linking the model with a web interface required debugging API calls.
   - *Solution*: Used Flask and Postman to streamline the deployment process.

---

## **Results and Visualizations**

1. **Confusion Matrix**: Displays per-class performance.
2. **Grad-CAM Visualization**: Highlights areas in the image critical for the model’s decision-making.

---

## **Future Enhancements**

- Incorporate attention mechanisms to further improve accuracy.
- Add a feedback loop for active learning to refine predictions.
- Extend the model to handle additional eye diseases.


