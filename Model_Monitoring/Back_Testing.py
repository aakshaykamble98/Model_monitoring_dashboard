import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import html
from io import BytesIO
import base64
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import plotly.graph_objects as go
import warnings
warnings.simplefilter("ignore")

def app():
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Back Testing</h1>
    """,
    unsafe_allow_html=True
)
    
    # Adding a small space between the title and the next section
    st.markdown(
        """
        <div style='margin-top: 15px; margin-bottom: 0px; text-align: center; font-size: 20px;'>
            <strong>K Fold Table</strong>
        </div>
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
            top: -37px;  /* Adjust to align vertically outside the table container */
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
    
    # Base directory of the current script
    base_dir = os.path.dirname(__file__)
    
    # Construct the full path to the CSV file
    csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
    
    # Read the CSV file
    Input_data = pd.read_csv(csv_file_path_1)
    
    # Base directory of the current script
    base_dir = os.path.dirname(__file__)
    
    # Construct the full path to the pickle file
    model_file_path = os.path.join(base_dir, 'pkl', 'Logistic.pkl')
    
    # Load the pre-trained RandomForest model
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
        
        
    # Input_data = wd.get_args("Input_data")
    
    # model = wd.get_args("Model")
    # print(model)

    col1, col2 = st.columns(2)

    with col1:
        No_Of_Fold = st.number_input("Enter the number of folds:", min_value=2, max_value=10, value=5, step=1)
    X= Input_data.drop('def_flag',axis=1)
    y = Input_data['def_flag']
    # train test split
    train_data,test_data,train_target,test_target=train_test_split(X,y,test_size=0.2,stratify=y)
    
    ## cross validation
    accuracy_scores = cross_val_score(model, train_data, train_target, cv=No_Of_Fold, scoring='accuracy')
    precision_scores = cross_val_score(model, train_data, train_target, cv=No_Of_Fold, scoring='precision')
    recall_scores = cross_val_score(model, train_data, train_target, cv=No_Of_Fold, scoring='recall')
    f1_scores = cross_val_score(model, train_data, train_target, cv=No_Of_Fold, scoring='f1')
    roc_scores = cross_val_score(model, train_data, train_target, cv=No_Of_Fold, scoring='roc_auc')
    
    temp = pd.DataFrame({"Score":['Minimum','Maximum','Mean'],
                                            " Accuracy": [min(accuracy_scores),max(accuracy_scores),np.mean(accuracy_scores)],
                                            " Precision": [min(precision_scores),max(precision_scores),np.mean(precision_scores)],
                                            " Recall": [min(recall_scores),max(recall_scores),np.mean(recall_scores)],
                                            " F1 score": [min(f1_scores),max(f1_scores),np.mean(f1_scores)],
                                            "ROC-AUC": [min(roc_scores),max(roc_scores),np.mean(roc_scores)]
                                            }).round(2)
    # wd.save_table(temp,"K Fold Table")
    # st.write('K Fold Table')
    # st.dataframe(temp, use_container_width=True)

    st.markdown("<div style='text-align: center; font-size: 16px;'></div>", unsafe_allow_html=True)

    # Base directory of the current script
    base_dir = os.path.dirname(__file__)

    # Construct the full path to the image file
    image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

    # df = df.drop(df.columns[0], axis=1)
    html_table = create_html_table_with_download(temp, "K_Fold_table.xlsx", image_path)
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Adding a small space between the title and the next section
    st.markdown(
        """
        <div style='margin-top: 15px; margin-bottom: 5px; text-align: center; font-size: 20px;'>
            <strong>ROC Curve for K Folds</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    kf = KFold(n_splits=No_Of_Fold, shuffle=True, random_state=42)
    
    fig = go.Figure()
    X, y = np.array(X), np.array(y)
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        probas_ = model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc_lr = auc(fpr, tpr)
        
        # Add ROC curve for the current fold
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f'ROC fold {fold + 1} (AUC = {roc_auc_lr:.2f})',
                line=dict(width=2)
            )
        )
    
    # Add No Skill line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="No Skill",
            line=dict(color="black", width=1, dash="dash"),
        )
    )
    
    # Update layout with centered title and axis labels
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=30, b=50)  # Set left, right, top, and bottom margins
    )
    
    # Plot the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app()