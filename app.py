import streamlit as st
import pickle
import os
import pandas as pd
from main import CustomClassifier
from visualize import visualize
import numpy as np
import altair as alt
st.set_page_config(layout="wide")

model_files = os.listdir('model')
selected_opt = st.sidebar.selectbox('Select feature', [ 'Visualize data', 'App'])




class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']



def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
def csv_input_feadures():
    file = st.sidebar.file_uploader("Upload CSV")
    if file is not None:
        try:
            df = pd.read_csv(file, header=None)
            df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        except:
            st.write('Please check your CSV file, make sure it has 4 columns: sepal_length, sepal_width, petal_length, petal_width')
            return None
        return df
    else:
        return None
def user_input(loaded_model):
    df = user_input_features()
    st.subheader('User Input parameters')
    st.write(df)
    y_pred = loaded_model.predict(df)
    print(y_pred)
    # prediction_proba = loaded_model.predict(df)
    st.subheader('Prediction')
    st.write(f'<span style="color: green; font-size:32px;">{class_name[y_pred[0]]}</span>', unsafe_allow_html=True)
    # hiện ảnh:

    image = f'images/{class_name[y_pred[0]]}.jpg'
    st.image(image, width=300)
def choose_image(x):
    url = None
    # print(x)
    if x==class_name[0]:
        url = 'https://i.ibb.co/yd3Lh8t/Iris-setosa.jpg'
    elif x==class_name[1]:
        url = 'https://i.ibb.co/pdR8TfX/Iris-versicolor.jpg'
    elif x==class_name[2]:
        url = 'https://i.ibb.co/6W6WvzS/Iris-virginica.jpg'
    return  f'![Image]({url})'
def csv_input(loaded_model):
    
    df = csv_input_feadures()
    if df is not None:
        df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        
        # st.write(df)
        y_pred = loaded_model.predict(df)
        df['Predicted class'] = [class_name[x] for x in y_pred]
        df['Image'] = df['Predicted class'].apply(lambda x: x)
        df['Image'] = df['Image'].apply(choose_image)
        
        # Display DataFrame with images embedded
        st.write(df.to_markdown(), unsafe_allow_html=True)
def __main__():
    if selected_opt == 'App':
        selected_model = st.sidebar.selectbox('Select Model', model_files)
        model_path = os.path.join('model', selected_model)
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        selected_type = st.sidebar.selectbox('Select Data Type', ['User Input', 'Input From CSV'])
        if selected_type == 'User Input':
            user_input(loaded_model)
        elif selected_type == 'Input From CSV':
            demo = st.sidebar.checkbox('Demo')
            if demo:
                df = pd.read_csv('data/X_test.csv')
                df = df.sample(5)
                df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                y_pred = loaded_model.predict(df)
                df['Predicted class'] = [class_name[x] for x in y_pred]
                df['Image'] = df['Predicted class'].apply(lambda x: x)
                df['Image'] = df['Image'].apply(choose_image)
                st.write(df.to_markdown(), unsafe_allow_html=True)
            else:
                csv_input(loaded_model)
    elif selected_opt == 'Visualize data':
        visualize()
        
if __name__ == '__main__':
    __main__()