import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# load model | scaler
def add_prediction(dictionary):
    model_path = os.path.join("models", "trained_model.joblib")
    scaler_path = os.path.join("scalers", "scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    input_array = np.array(list(dictionary.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    
    st.write("")
    st.write("")
    st.write("Cell clusture prediction")
    st.write("The cell clusture is")
    if prediction == 0:
        st.write("Bening")
    else:
        st.write("Malicious")

    st.write("Probability of being bening: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("While this app can aid medical professionals in diagnosis, it must not be utilized as a replacement for professional diagnosis and medical advice.")
    




# get data same way
def get_data():
    data = pd.read_csv("data/data.csv") #read the data
    
    # we are going to drop "id" & "Unnamed: 32" columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Converting existing column "diagnosis" into binary
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x=="M" else 0)

    return data



# key:value normalize 0<-->1
def normalise_dict_values(dictionary):
    data = get_data()
    
    x = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    for key, value in dictionary.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scale_val = (value-min_val) / (max_val-min_val)
        scaled_dict[key] = scale_val
    
    return scaled_dict



# sidebar | components
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_data()
    # column name --> lebel
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # contain data column name: column value
    columnValue = {}
    
    for label, key in slider_labels:
        columnValue[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value= float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return columnValue



# plotly
def get_rader_chart(input_data):
    input_data = normalise_dict_values(input_data)
    categories = ['Radius', 'Texture', 
                'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=False
    )
    
    return fig



def main():
    
    st.set_page_config(
        page_title="Risk Analysis for Breast Cancer",
        page_icon="Female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

        
    column_data = add_sidebar()  #data[key:value] --> {ColumnName: ColumnValue}
    
    with st.container():
        st.title("Predictive Breast Cancer Analysis")
        st.write("Experience the potential of machine learning in aiding diagnosis! Integrate this empowering app with your cytology lab. Together, we'll enhance breast cancer diagnosis from tissue samples, leveraging cutting-edge technology to deliver more accurate and informed insights for patients. Witness the power of our model in action!")
    
    
    
    col1, col2 = st.columns([4, 1])
    with col1:
        rader_chart = get_rader_chart(column_data)
        st.plotly_chart(rader_chart)
    with col2:
        add_prediction(column_data)
        
    

    st.write("Hello World")
    st.write("Made with ❤")
    st.write("-----------")

if __name__ == '__main__':
    main()