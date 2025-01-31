import os

import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from catboost import CatBoostClassifier













from ml_utility import (read_data,
                        preprocess_data,
                        train_model,
                        evaluate_model)


working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)


st.set_page_config(
    page_title="Automate ML",
    page_icon="🧠",
    layout="centered")


st.title("🤖 No Code ML Model Training")

dataset_list = os.listdir(f"{parent_dir}/data")

dataset = st.selectbox("Select a dataset from the dropdown",
                       dataset_list,
                       index=None)

df = read_data(dataset)

if df is not None:
    st.dataframe(df.head())

    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ["standard", "minmax"]

    model_dictionary = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Extra Trees Classifier": ExtraTreesClassifier(),
        "Multi-Layer Perceptron Classifier": MLPClassifier(),
        "Bagging Classifier": BaggingClassifier(),
        "CatBoost Classifier": CatBoostClassifier(verbose=0)  # Silent mode
    }


    with col1:
        target_column = st.selectbox("Select the Target Column", list(df.columns))
    with col2:
        scaler_type = st.selectbox("Select a scaler", scaler_type_list)
    with col3:
        selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
    with col4:
        model_name = st.text_input("Model name")

    if st.button("Train the Model"):

        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

        model_to_be_trained = model_dictionary[selected_model]

        model = train_model(X_train, y_train, model_to_be_trained, model_name)

        accuracy = evaluate_model(model, X_test, y_test)

        st.success("Test Accuracy: " + str(accuracy))