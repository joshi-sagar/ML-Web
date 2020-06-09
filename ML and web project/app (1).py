import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification WEb App.")
    st.sidebar.title("Binary Classification Web App.")
    st.markdown("Are your mushroom edible or poisionous? 🍄")
    st.sidebar.markdown("Are your mushroom edible or poisionous? 🍄")

    @st.cache(persist= True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    @st.cache(persist= True)
    def split(df):
        y= df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.3, random_state =0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics):
        if 'Confusion Martix' in metrics:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels= class_names)
            st.pyplot()

        if 'ROC Curve' in metrics:
            st.subheader("ROC curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall' in metrics:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

   
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameters)", 0.01, 10.0, step =0.01, key ='C')
        Kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'Kernel')
        gamma = st.sidebar.radio("Gamma(Kernel coefficient)",("scale", "auto"), key = 'gamma')

        metrics = st.sidebar.multiselect("What metrics to Plot ?", ('Confusion Martix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=Kernel, gamma = gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Pricision : ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
# Logistic Regression
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameters)", 0.01, 10.0, step =0.01, key ='C_LR')
        max_iter = st.sidebar.slider("maximum number of interations", 100, 500, key = 'max_iter')

        metrics = st.sidebar.multiselect("What metrics to Plot ?", ('Confusion Martix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter= max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Pricision : ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)
# Random Forest
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input ("The number of trees in the forest",100, 5000, step = 10 ,key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1 , 20, step= 1, key = 'max_depth')
        bootstrap = st.sidebar.radio ("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to Plot ?", ('Confusion Martix', 'ROC Curve', 'Precision-Recall'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Rsults")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth, bootstrap= bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Pricision : ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data.",False):
        st.subheader("Mushroom Data set (Classification)")
        st.write(df)

    
if __name__ == '__main__':
    main()


