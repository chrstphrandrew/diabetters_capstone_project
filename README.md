# Predicting Diabetes with Machine Learning and Streamlit Dashboard

###
This repository combines a machine learning model for predicting diabetes with a Streamlit dashboard for user interaction and visualization. Below is an overview of the key components and functionalities of the project.
###


# Machine Learning Model
# Dataset:
The dataset used is the Pima Indians Diabetes dataset, consisting of medical predictor variables and an outcome variable indicating diabetes presence.


# Preprocessing:
Features are standardized using StandardScaler to ensure all features contribute equally to the model.

# Model Architecture:
The neural network model is built using Keras with tunable hyperparameters:
    1. Multiple dense layers with ReLU activation and batch normalization.
    2. Dropout layers to prevent overfitting.
    3. Output layer with sigmoid activation for binary classification.

# Hyperparameter Tuning:
Hyperparameters such as layer units, dropout rates, and learning rate are optimized using RandomSearch from Keras Tuner.

K-Fold cross-validation (K=5) is employed to evaluate model performance robustly.
Training and Evaluation:

The best model configuration based on validation accuracy is selected and saved.

Training includes early stopping and learning rate reduction to improve convergence and prevent overfitting.

Evaluation metrics include accuracy score to assess the model's predictive performance.

# Streamlit Dashboard
# Features:

#   1. Home: 
Overview of the application and its purpose.
#   2. Make a Prediction: 
User-friendly interface to input health parameters and obtain diabetes prediction.
#   3. About Diabetes: 
Information on diabetes types and the Pima Indians Diabetes Dataset.

# Implementation:

# Sidebar Navigation: 
Allows users to switch between different sections of the dashboard.
# Prediction: 
Takes user inputs, preprocesses them with a pre-trained scaler, and uses a pre-trained Keras model to predict diabetes likelihood.
# Visualization: 
Displays prediction results and input summary dynamically with appropriate messages and images based on model predictions.

# To run the application:
    1.Clone this repository and install necessary libraries (numpy, pandas, scikit-learn, keras,       kerastuner, streamlit).
    2.Ensure Python environment compatibility (recommended version: Python 3.7+).
    Execute the Streamlit app with streamlit run your_script_name.py.
    3.Interact with the dashboard via the provided interface to explore diabetes prediction functionality.

# Conclusion
This project demonstrates the integration of machine learning with user-friendly web applications using Streamlit. By leveraging advanced techniques in neural networks and hyperparameter optimization, the model provides accurate predictions based on health data. The inclusion of a dashboard enhances usability and accessibility, making it suitable for both research and practical applications in healthcare.

For further information or contributions, please contact the project developer.