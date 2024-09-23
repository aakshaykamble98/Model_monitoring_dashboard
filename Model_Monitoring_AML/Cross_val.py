import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import html
from io import BytesIO
import base64
from sklearn.model_selection import cross_val_score
import warnings
warnings.simplefilter("ignore")

def app():
    # Centered text in the floating container
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Cross Validation</h1>
        """,
        unsafe_allow_html=True
    )

    #Fucntion for creating HTML table and download link to downnload the table
    def create_html_table_with_download(dataframe, file_name, image_path):

        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode()

        # Create download link with hover and clicked effect
        download_link = f"""
        <style>
            .download-icon img {{
                width: 70px;
                height: 70px;
                border-radius: 30%;  /* Makes the image circular */
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .download-icon img:hover {{
                transform: scale(1.1);  /* Slight zoom effect on hover */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);  /* Shadow effect */
            }}
            .download-icon img:active {{
                transform: scale(0.90);  /* Slight shrink effect on click */
                box-shadow: none;  /* Remove shadow when clicked */
            }}
        </style>

        <div class="download-icon">
            <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(to_excel(dataframe)).decode()}" download="{file_name}" title="Click to download the file">
                <img src="data:image/png;base64,{image_base64}" alt="Download Image" style="width:27px; height:auto;">
            </a>
        </div>
        """

        html_table = f"""
        <div class="custom-container">
            <table class="dataframe">
                <thead><tr>
        """
        for col_name in dataframe.columns:
            html_table += f'<th>{html.escape(col_name)}</th>'
        html_table += '</tr></thead><tbody>'
        for _, row in dataframe.iterrows():
            html_table += '<tr>'
            for col_value in row:
                html_table += f'<td>{html.escape(str(col_value))}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table></div>'

        # Combine the download link and the table
        return download_link + html_table

    # Function to convert DataFrame to Excel
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data

    custom_css = """
    <style>
        .custom-container {
            max-height: 400px;
            max-width: 100%;
            overflow-y: scroll;
            overflow-x: scroll;
            position: relative;
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        table {
            width: 100%;
            height: auto;
        }
        th, td {
            font-size: 14px;
            padding: 8px;
            text-align: left;
            white-space: nowrap;  /* Prevent text from wrapping */

        }
        .download-icon {
            position: absolute;
            right: 40px;
            top: -40px;  /* Adjust to align vertically outside the table container */
            font-size: 24px;   
        }
        thead {
            position: sticky;
            top: 0;
            background-color: rgb(39, 45, 85);
            color: rgb(20, 26, 63);
        }
        thead th {
            color: white;
            font-weight: normal;
        }
        .info-container {
            margin-top: 10px;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f9f9f9;
            font-size: 14px;
            text-align: left;
            color: black; 
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Adding a small space between the title and the next section
    st.markdown(
        """
        <div style='margin-bottom: 2px; text-align: center; font-size: 20px;'>
            <strong>K Cross validation Summary</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Base directory of the current script
    base_dir = os.path.dirname(__file__)
    
    # Construct the full path to the CSV file
    csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
    csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Monitoring_Data.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path_1)
    df1 = pd.read_csv(csv_file_path_2)
    
    # Base directory of the current script
    base_dir = os.path.dirname(__file__)
    
    # Construct the full path to the pickle file
    model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
    
    # Load the pre-trained RandomForest model
    with open(model_file_path, 'rb') as file:
        Model = pickle.load(file)
    
    Target_Variable = 'ALERT_STATUS'
    
    CV = 5

    # Calculate cross-validation scores for the Development data
    scores_dev = cross_val_score(Model, df.drop(columns=[Target_Variable]), df[Target_Variable], cv=CV)
    
    # Create a summary DataFrame for the Development data scores
    summary_dev = pd.DataFrame(scores_dev, columns=["Development_K-Cross_Accuracy"]).describe().round(2)
    
    # Calculate cross-validation scores for the Monitoring data
    scores_mon = cross_val_score(Model, df1.drop(columns=[Target_Variable]), df1[Target_Variable], cv=CV)
    
    # Create a DataFrame for the Monitoring data scores and calculate summary statistics
    summary_mon = pd.DataFrame(scores_mon, columns=["Monitoring_K-Cross_Accuracy"]).describe().round(2)
    
    # Concatenate the summaries horizontally
    summary = pd.concat([summary_dev, summary_mon], axis=1)
    
    # Rename the index to "Measures"
    summary.index.name = "Measures"
    
    # st.dataframe(summary, use_container_width=True)

    # Assuming df is your DataFrame
    summary = summary.reset_index()  # Reset the index and make it a column
    summary.rename(columns={'index': 'Measure'}, inplace=True)

    # Base directory of the current script
    base_dir = os.path.dirname(__file__)

    # Construct the full path to the image file
    image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

    html_table = create_html_table_with_download(summary, "VIF_table.xlsx", image_path)
    st.markdown(html_table, unsafe_allow_html=True)

if __name__ == "__main__":
    app()