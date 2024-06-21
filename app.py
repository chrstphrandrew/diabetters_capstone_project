import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image

# Load the trained model
model = load_model('diabetters.h5')

# Load the scaler (you might need to save the scaler during training and load it here)
# Here, I'm recreating the scaler assuming the training scaler
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dataset = pd.read_csv(url, header=None, names=column_names)

# Prepare scaler
X = dataset.iloc[:, :-1].values
scaler = StandardScaler()
scaler.fit(X)

# Sidebar for navigation
st.sidebar.title("Diabetters Menu")
menu = st.sidebar.radio("choose the pages you want", ["Home", "Make a Prediction", "About Diabetes"])

if menu == "Home":
    # Home page
    st.title("Diabetters")
    st.write("### User friendly Diabetes Prediction Website")
    st.write("powered by Tensorflow and Streamlit")
    st.image(Image.open('diabetes.png'), caption='By The Health News Team', use_column_width=True)
    st.write("""
    ## Welcome to Diabetters!
    This application predicts the likelihood of having diabetes based on various health parameters.
    Use the diabetters menu on the sidebar to proceed to the prediction section or to learn more about diabetes.
    """)
elif menu == "Make a Prediction":
    # Prediction page
    st.title("Make a Prediction")
    st.write("Press the button if the input already filled")

    # Input fields in the sidebar
    st.sidebar.markdown("Enter the following details to predict diabetes:")
    pregnancies = st.sidebar.slider('Pregnancies', min_value=0, max_value=20, value=0, step=1)
    glucose = st.sidebar.slider('Glucose', min_value=0, max_value=200, value=0, step=1)
    blood_pressure = st.sidebar.slider('Blood Pressure', min_value=0, max_value=150, value=0, step=1)
    skin_thickness = st.sidebar.slider('Skin Thickness', min_value=0, max_value=100, value=0, step=1)
    insulin = st.sidebar.slider('Insulin', min_value=0, max_value=900, value=0, step=1)
    bmi = st.sidebar.slider('BMI', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0, step=0.01)
    age = st.sidebar.slider('Age', min_value=0, max_value=120, value=0, step=1)

    # Button to predict
    if st.button('Predict'):
        # Prepare the input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = (prediction > 0.5).astype("int32")

        # Display the result
        st.subheader('Prediction Result')
        if predicted_class[0][0] == 1:
            st.error('The model predicts that this person has diabetes.')
            st.image('diabetes_positive.gif', use_column_width=True)
        else:
            st.success('The model predicts that this person does not have diabetes.')
            st.image('diabetes_negative.gif', use_column_width=True)

        # Summary of input data
        st.subheader('Input Summary')
        input_summary = f"""
        - **Pregnancies:** {pregnancies}
        - **Glucose:** {glucose}
        - **Blood Pressure:** {blood_pressure}
        - **Skin Thickness:** {skin_thickness}
        - **Insulin:** {insulin}
        - **BMI:** {bmi}
        - **Diabetes Pedigree Function:** {diabetes_pedigree_function}
        - **Age:** {age}
        """
        st.markdown(input_summary)
elif menu == "About Diabetes":
    # About Diabetes page
    st.title("Diabetes Information")

    st.image('diabetes_info.jpg', caption='Photo by Vector Stock', use_column_width=True)

    st.write("""
    ## What is Diabetes?
    Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
    There are three main types of diabetes: type 1, type 2, and gestational diabetes (diabetes while pregnant).
    ### Type 1 Diabetes
    Type 1 diabetes is thought to be caused by an autoimmune reaction (the body attacks itself by mistake) that stops your body from making insulin.
    ### Type 2 Diabetes
    With type 2 diabetes, your body doesn’t use insulin well and can’t keep blood sugar at normal levels.
    ### Gestational Diabetes
    Gestational diabetes develops in pregnant women who have never had diabetes.
    """)
    st.write("""
    ## Pima Indians Diabetes Dataset
    The Pima Indians Diabetes Dataset is used to predict diabetes. This dataset contains several medical predictor variables and one target variable, Outcome. Here are the key features:

    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration (a two-hour oral glucose tolerance test)
    - **Blood Pressure**: Diastolic blood pressure (mm Hg)
    - **Skin Thickness**: Triceps skinfold thickness (mm)
    - **Insulin**: 2-Hour serum insulin (mu U/ml)
    - **BMI**: Body mass index (weight in kg/(height in m)^2)
    - **Diabetes Pedigree Function**: A function which scores likelihood of diabetes based on family history
    - **Age**: Age (years)

    This dataset helps in building a model to predict whether a person is likely to have diabetes based on diagnostic measurements.
    """)
    

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Christopher Andrew 'Diabetters'")
