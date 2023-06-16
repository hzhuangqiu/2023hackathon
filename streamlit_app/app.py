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
    'Water Properties Group 1': ['pH', 'Color', 'Conductivity'],
    'Water Properties Group 2': ['Odor', 'Turbidity', 'Total Dissolved Solids'],
    'Mineral Elements Group 1': ['Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc'],
    'Mineral Elements Group 2': ['Fluoride', 'Copper', 'Sulfate', 'Chlorine', 'Manganese'],
    'Temperature': ['Water Temperature', 'Air Temperature'],
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

# Predicting function
def predict_fresh_water_quality(input_data):
    # Initialize empty list to store results
    results = []
    
    # Iterate through each row in input DataFrame
    for index, row in input_data.iterrows():
        # Reshape input features and make prediction
        input_features = np.array(row).reshape(1, -1)
        prediction = model.predict(input_features)[0]
        
        # Map prediction to quality label and append to results list
        quality_labels = {0: "Bad", 1: "Good"}
        quality = quality_labels[prediction]
        results.append(quality)

    return results

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

    st.subheader("2. Distribution of pH levels in Data")
    phexpander = st.expander("🔎 Click here to view the graph")
    with phexpander:
        ph_data=data[['pH']]
        st.line_chart(ph_data)

    st.subheader("3. Distribution of mineral elements in water")
    distributionexpander = st.expander("🔎 Click here to view the graph")
    with distributionexpander:
        chart_data=data[['Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Odor', 'Chlorine', 'Manganese']]
        st.bar_chart(chart_data)

    st.subheader("4. Distribution of water properties")
    propertyexpander = st.expander("🔎 Click here to view the graph")
    with propertyexpander:
        prop_data=data[['Color','Conductivity','Total Dissolved Solids']]
        st.area_chart(prop_data)

    st.subheader("5. Temperature Distribution")
    tempexpander = st.expander("🔎 Click here to view the graph")
    with tempexpander :
        prop_data=data[['Water Temperature','Air Temperature']]
        st.line_chart(prop_data)

def output(input_data, predict_result, output_mode='Batched'):
    
    input_df = pd.DataFrame(input_data)
    quality_df = pd.DataFrame({'Quality': predict_result})
    combined_df = pd.concat([quality_df, input_df], axis=1)
    
    st.subheader("Output 💧")
    csv = convert_df_to_csv(combined_df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Output.csv',
        mime='text/csv')

    st.write("The processed data is : ")
    st.table(combined_df) 

    if output_mode == 'Batched':
        output_overview(combined_df)
    elif output_mode == 'Single':
        return
    else:
        return not_implement_error()

def csv_uploader():
    uploaded_file = st.file_uploader('Upload your CSV file', type='csv')
    data = []
    if uploaded_file is not None:
        data = read_csv(uploaded_file)
    return data

def user_choose_model():
    model_name = st.selectbox('Choose your model', ('LGBM', 'XGBoost', 'CatBoost'))
    model = backend_load_model(model_name)
    return model

def normalize_data(input_data):
    data_df = pd.DataFrame(input_data)
    return data_df

def user_train():
    return not_implement_error()
    model = user_choose_model()
    data = csv_uploader()
    
    backend_train(model, data)
    
    return

def user_predict():
    model = user_choose_model()
    if not model:
        return not_implement_error()
    data = []
    
    predict_mode = st.radio('Choose your predict mode', ('Single', 'Batched'))
    if predict_mode == 'Batched':
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
                print(input_data)
                data = normalize_data(input_data)
    else:
        return not_implement_error()
    
    output_mode = predict_mode
    if len(data)>0:
        predict_result = backend_predict(model, data)
        output(data, predict_result, output_mode)
    
    return

# Define Streamlit app
def main():
    st.title('💧 Fresh Water Quality Detector (FWD)')

    run_mode = st.selectbox('Choose your run mode', ('Predict', 'Train (Not Impl yet)'))
    
    if run_mode == 'Predict':
        user_predict()
    else:
        return not_implement_error()
   
#     # Display sample template for user to download
#     sample_template = pd.DataFrame(default_values, index=[0])
#      # Converting to CSV as downloadable button
#     template_content = convert_df(sample_template)
#     st.download_button(
#     label="Click here to download the template file",
#     data=template_content,
#     file_name='Sample-template.csv',
#     mime='text/csv')

#     # Read uploaded CSV file
#     if uploaded_file is not None:
#         test_data = read_csv(uploaded_file)
#         expander = st.expander("🔎 Click to view uploaded file content")
#         with expander:
#             st.write(test_data)
#             # Create a submit button
#         submit = st.button("Check  Quality 🔬",type="primary")

#         # If the submit button is clicked, make the prediction
#         if submit:
#             process_data=processed_data(test_data)
#             quality_checker=predict_fresh_water_quality(process_data)
#             output(process_data,quality_checker)
#             # Print the prediction
#             st.balloons()
#     else:
#         st.info('Download the above template and fill in your data.')
#         # Set page footer 
#     st.write("\n\nMade with :heart: by Team Humanoids 🤖")
#     st.write("Intel® oneAPI Hackathon for Open Innovation 2023.")

if __name__ == '__main__':
    main()
    