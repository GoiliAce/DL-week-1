import altair as alt
import streamlit as st
import pandas as pd
import pickle
import os

def histogram4(source):
    base = alt.Chart(source)

    hist_petal_width = base.mark_bar().encode(
        alt.X('petal_width:Q', bin=alt.Bin(maxbins=20), title='Petal Width'),
        alt.Y('count():Q', title='Count'),
        alt.Color(value='#e48f4d')
    ).properties(width=250, height=170)

    hist_petal_length = base.mark_bar().encode(
        alt.X('petal_length:Q', bin=alt.Bin(maxbins=20), title='Petal Length'),
        alt.Y('count():Q', title='Count'),
        alt.Color(value='#5fbb9f')
    ).properties(width=250, height=170)

    hist_sepal_width = base.mark_bar().encode(
        alt.X('sepal_width:Q', bin=alt.Bin(maxbins=20), title='Sepal Width'),
        alt.Y('count():Q', title='Count'),
        alt.Color(value='#ee69ad')
    ).properties(width=250, height=170)

    hist_sepal_length = base.mark_bar().encode(
        alt.X('sepal_length:Q', bin=alt.Bin(maxbins=20), title='Sepal Length'),
        alt.Y('count():Q', title='Count'),
        alt.Color(value='#9e9ac9')
    ).properties(width=250, height=170)
    chart =  alt.hconcat(hist_petal_width, hist_petal_length, hist_sepal_width, hist_sepal_length)
    st.altair_chart(chart, use_container_width=True)
    chart = (
        alt.Chart(source).transform_fold(
            ['petal_width', 'petal_length', 'sepal_width', 'sepal_length'],
            as_=['Measurement_type', 'value']
        ).mark_bar(
            opacity=0.7,
            binSpacing=0
        ).encode(
            alt.X('value:Q', bin=alt.Bin(maxbins=50)),
            alt.Y('count():Q'),
            alt.Color('Measurement_type:N').scale(scheme='dark2')
        )
    )
    st.altair_chart(chart, use_container_width=True)
    
def donut_chart(source):
    class_counts = source['class'].value_counts().reset_index()
    class_counts.columns = ['class', 'count']
    
    chart = alt.Chart(class_counts).mark_arc(innerRadius=50).encode(
        theta="count",
        color=alt.Color("class:N").scale(scheme='accent')
    )
    
    st.altair_chart(chart, use_container_width=True)
# visualize using kmeans
def visualize_model():
    model_files = os.listdir('model')
    selected_model = st.sidebar.selectbox('Select Model', model_files)
    model_path = os.path.join('model', selected_model)
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    data_test = pd.read_csv('data/X_test.csv')
    data_test.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data_test['cluster'] = loaded_model.predict(data_test)
    x = st.sidebar.selectbox('X', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = st.sidebar.selectbox('Y', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    chart = alt.Chart(data_test).mark_circle(size=100).encode(
        x=x,
        y=y,        
        color=alt.Color('cluster:N').scale(scheme='accent'),
        tooltip=['cluster', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
def his_class(source):
    st.subheader('Phân tích bộ dữ liệu theo từng lớp qua các biểu đồ.')
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    boxplot_charts = []
    st.write('Biểu diễn phân bổ data theo class bằng boxplot')
        # Loop through columns and create boxplot charts
    for col in columns:
        boxplot_chart = alt.Chart(source).mark_boxplot(size=50).encode(
            x=alt.X('class:N', axis=None),
            y=col,
            color=alt.Color('class:N').scale(scheme='accent')
        ).properties(
            width=250,
            height=200
        )
        boxplot_charts.append(boxplot_chart)
    combined_chart = alt.hconcat(*boxplot_charts)
    st.altair_chart(combined_chart, use_container_width=True)
    st.write('Biểu diễn phân bổ data theo class bằng histogram')
    
    hist_charts = []

        # Loop through columns and create histogram charts
    for col in columns:
        hist_chart = alt.Chart(source).mark_bar().encode(
            alt.X(col, bin=alt.Bin(maxbins=20)),
            alt.Y('count()', title='Count'),
            alt.Color('class:N').scale(scheme='accent')
        ).properties(
            width=250,
            height=200
        )
        hist_charts.append(hist_chart)
    combined_chart = alt.hconcat(*hist_charts)

    # Show the combined chart
    st.altair_chart(combined_chart, use_container_width=True)
    st.write('Biểu diễn phân bổ data theo class bằng scatter')
    x = st.sidebar.selectbox('X', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = st.sidebar.selectbox('Y', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    chart = alt.Chart(source).mark_circle(size=70).encode(
        x=x,
        y=y,
        color= alt.Color('class:N').scale(scheme='accent'),        
        tooltip=['class', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
# Show the combined chart
def visualize():
    st.write('# Visualize data')
    source = pd.read_csv('data/iris/iris.data', header=None)
    source.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    select_option = st.sidebar.selectbox('Select Chart', ['Trực quan hóa dữ liệu với các biểu đồ box, histogram', 'Phân tích bộ dữ liệu theo từng lớp qua các biểu đồ.', 'Phân bố dữ liệu class', 'Phân bổ dữ liệu test với model đã train'])
    if select_option == 'Trực quan hóa dữ liệu với các biểu đồ box, histogram':
        histogram4(source)
    elif select_option == 'Phân bố dữ liệu class':
        donut_chart(source)
    elif select_option == 'Phân bổ dữ liệu test với model đã train':
        visualize_model()
    elif select_option == 'Phân tích bộ dữ liệu theo từng lớp qua các biểu đồ.':
        his_class(source)
    # histogram4(source)
    # donut_chart(source)
    # visualize_model()
    # his_class(source)