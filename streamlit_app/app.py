import streamlit as st
import pandas as pd
import base64
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from backend_wrapper import *

# Define value groups
value_groups = {
    'Water Properties Group 1': ['pH', 'Color', 'Odor', 'Turbidity'],
    'Water Properties Group 2': ['Conductivity', 'Total Dissolved Solids'],
    'Mineral Elements Group 1': ['Chloride', 'Sulfate'],
    'Mineral Elements Group 2': ['Nitrate', 'Zinc', 'Chlorine'],
    'Mineral Elements Group 3': ['Iron', 'Fluoride', 'Copper'],
    'Mineral Elements Group 4': ['Lead', 'Manganese'],
    'Temperature': ['Air Temperature', 'Water Temperature'],
}

# Define default values
default_values = {
    'pH': 7.0,
    'Iron': 0.0,
    'Nitrate': 0.0,
    'Chloride': 0.0,
    'Lead': 0.0,
    'Zinc': 0.0,
    'Color': 0,
    'Turbidity': 0.0,
    'Fluoride': 0.0,
    'Copper': 0.0,
    'Odor': 0.0,
    'Sulfate': 0.0,
    'Conductivity': 0.0,
    'Chlorine': 0.0,
    'Manganese': 0.0,
    'Total Dissolved Solids': 0.0,
    'Water Temperature': 0.0,
    'Air Temperature': 0.0
}

def read_csv(file):
    df = pd.read_csv(file)
    return df

def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

def output_overview(data):
    st.write("---")
    st.header("Data Overview")

    st.subheader("1. Output Overview")
    piechartexpander = st.expander("🔎 Click here to view the graph")
    with piechartexpander :
        quality_counts = data['Quality'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        ax.set_title('Distribution of Output Results in Dataset')
        st.pyplot(fig)

    water_properties_overview_col = st.columns(2)

    mineral_elements_overview_col_1 = st.columns(2)
    mineral_elements_overview_col_2 = st.columns(2)

    water_properties_overview_col[0].subheader("2.1. Distribution of water properties group 1")
    propertyexpander = st.expander("🔎 Click here to view the graph")
    with propertyexpander:
        prop_data=data[['pH', 'Color', 'Odor', 'Turbidity']]
        water_properties_overview_col[0].area_chart(prop_data)

    water_properties_overview_col[1].subheader("2.2. Distribution of water properties group 2")
    propertyexpander = st.expander("🔎 Click here to view the graph")
    with propertyexpander:
        prop_data=data[['Conductivity', 'Total Dissolved Solids']]
        water_properties_overview_col[1].area_chart(prop_data)

    mineral_elements_overview_col_1[0].subheader("3.1. Distribution of mineral elements in water")
    distributionexpander = st.expander("🔎 Click here to view the graph")
    with distributionexpander:
        chart_data=data[['Chloride', 'Sulfate']]
        mineral_elements_overview_col_1[0].bar_chart(chart_data)

    mineral_elements_overview_col_1[1].subheader("3.2. Distribution of mineral elements in water")
    distributionexpander = st.expander("🔎 Click here to view the graph")
    with distributionexpander:
        chart_data=data[['Nitrate', 'Zinc', 'Chlorine']]
        mineral_elements_overview_col_1[1].bar_chart(chart_data)

    mineral_elements_overview_col_2[0].subheader("3.3. Distribution of mineral elements in water")
    distributionexpander = st.expander("🔎 Click here to view the graph")
    with distributionexpander:
        chart_data=data[['Iron', 'Fluoride', 'Copper']]
        mineral_elements_overview_col_2[0].bar_chart(chart_data)

    mineral_elements_overview_col_2[1].subheader("3.4. Distribution of mineral elements in water")
    distributionexpander = st.expander("🔎 Click here to view the graph")
    with distributionexpander:
        chart_data=data[['Lead', 'Manganese']]
        mineral_elements_overview_col_2[1].bar_chart(chart_data)

    st.subheader("4. Temperature Distribution")
    tempexpander = st.expander("🔎 Click here to view the graph")
    with tempexpander :
        prop_data=data[['Air Temperature', 'Water Temperature']]
        st.area_chart(prop_data)

def output(input_data, predict_result, predict_time, output_mode='Batched'):
    
    input_df = pd.DataFrame(input_data)
    quality_df = pd.DataFrame({'Quality': predict_result})
    combined_df = pd.concat([quality_df, input_df], axis=1)
    
    st.subheader("Output 💧")
    csv = convert_df_to_csv(combined_df)
    output_info_col = st.columns(3)
    output_info_col[0].write("Predict time: %.3f ms" % predict_time)
    output_info_col[1].download_button(
        label="Download data as CSV",
        data=csv,
        file_name='output.csv',
        mime='text/csv')

    with st.expander("🔎 Click here to view details"):
        st.table(combined_df) 

    if output_mode == 'Batched':
        output_overview(combined_df)
    elif output_mode == 'Single':
        return
    else:
        return not_implement_error()

def csv_uploader():
    data = []
    with st.form('single line input'):
        uploaded_file = st.file_uploader('Upload your CSV file', type='csv')
        submitted = st.form_submit_button('Submit')
        if submitted and uploaded_file is not None:
            data = read_csv(uploaded_file)
            data = normalize_data(data)
    return data

def user_choose_model():
    model_name = st.selectbox('Choose your model', ('LGBM', 'XGBoost', 'CatBoost'))
    model = backend_load_model(model_name)
    return model

def normalize_data(input_data):
    data_df = pd.DataFrame(input_data)
    data_df = data_df.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1, errors='ignore')
    data_df = data_df.drop("Target", axis=1, errors='ignore')
    return data_df

def sample_csv_downloader():
    sample_template = pd.DataFrame(default_values, index=[0])
    template_content = convert_df_to_csv(sample_template)
    st.download_button(
        label="Click here to download the template CSV file",
        data=template_content,
        file_name='sample_template.csv',
        mime='text/csv')
    return

def user_train():
    return not_implement_error()
    model = user_choose_model()
    sample_csv_downloader()
    data = csv_uploader()
    backend_train(model, data)
    return

def user_predict():
    model = user_choose_model()
    if not model:
        return not_implement_error()

    data = []
    predict_mode_map = {'Single instance input': 'Single', 'Batched instance input (upload CSV file)': 'Batched'}
    predict_mode_choice = st.radio('Choose your predict mode', predict_mode_map.keys())
    predict_mode = predict_mode_map[predict_mode_choice]
    if predict_mode == 'Batched':
        sample_csv_downloader()
        data = csv_uploader()
    elif predict_mode == 'Single':
        with st.form('single line input'):
            input_data = {}
            for group_key, group_val in value_groups.items():
                columns = st.columns(len(group_val))
                for idx, col in enumerate(columns):
                    input_name = group_val[idx]
                    input_data[input_name] = [float(col.text_input(input_name, default_values[input_name]))]
            submitted = st.form_submit_button('Submit')
            if submitted:
                data = normalize_data(input_data)
    else:
        return not_implement_error()
    
    output_mode = predict_mode
    if len(data)>0:
        predict_time, predict_result = backend_predict(model, data)
        output(data, predict_result, predict_time, output_mode)
    
    return

# Define Streamlit app
def main():
    st.title('💧 Fresh Water Quality Predictor')

    # only support predict so far
    # run_mode = st.selectbox('Choose your run mode', ('Predict', 'Train (Not Impl yet)'))
    run_mode = 'Predict'
    
    if run_mode == 'Predict':
        user_predict()
    else:
        return not_implement_error()

if __name__ == '__main__':
    main()
    