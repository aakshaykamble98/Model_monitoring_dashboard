import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import html
from io import BytesIO
import base64
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import warnings 
warnings.filterwarnings(action='ignore')


# Streamlit app
def app():
    # Centered text in the floating container
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Model Monitoring</h1>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(['Tables', 'Graphs'])
    
    with tab1:
        # Adding a small space between the title and the next section
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Model Monitoring</strong>
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
        
        # Base directory of the current script
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the CSV file
        csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path_2)
        
        # Base directory of the current script
        base_dir = os.path.dirname(__file__)

        # Base directory of the current script
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the pickle file
        model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
        
        # Load the pre-trained RandomForest model
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
        
        # Assuming the target variable is named 'target'
        target = 'ALERT_STATUS'
        
        # your code here
        test_size=0.2
        if "pk"in df.columns :
            df.drop("pk",axis=1,inplace=True)
        # X=df[df.columns[df.columns!=target.columns[0]]]
        # Y=np.array(target.values)
        X = df.drop(columns=[target])
        Y = df[target].values
        Y=Y.reshape(Y.shape[0])

        #RandomForestClassifier
        # model=LogisticRegression()
        train_data, test_data, train_target, test_target = train_test_split(X, Y, test_size=test_size, stratify=Y)
        
        train=pd.DataFrame(train_target)[0].value_counts()
        
        #Classes wise Frequency of Train
        train = pd.DataFrame({"Class":list(train.index.values),"Development_Count (Train Data)": list(train.values)})
        test = pd.DataFrame(test_target)[0].value_counts()
        test = pd.DataFrame({"Class":list(test.index.values),"Monitoring_Count (Test Data)": list(test.values)})
        train_test = pd.merge(train, test)
        
        # wd.save_table(train_test,name="Count Table")
        st.markdown("<div style='text-align: center; font-size: 15px;'>Count Table</div>", unsafe_allow_html=True)
        # st.dataframe(train_test, use_container_width=True)

        # Base directory of the current script
        base_dir = os.path.dirname(__file__)

        # Construct the full path to the image file
        image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

        # df = df.drop(df.columns[0], axis=1)
        html_table = create_html_table_with_download(train_test, "Count_Table.xlsx", image_path)
        st.markdown(html_table, unsafe_allow_html=True)
        
        #Classes wise Frequency
        model.fit(train_data,train_target)
        y_pred=model.predict(train_data)
        y_pred_test=model.predict(test_data)
        
        # Compute confusion matrices
        cm_train = sk_confusion_matrix(train_target, y_pred)
        cm_test = sk_confusion_matrix(test_target, y_pred_test)
        
        # Create formatted confusion matrices
        formatted_cm_train = pd.DataFrame(
            {
                'Predicted Negative': [f'TN: {cm_train[0, 0]}', f'FP: {cm_train[0, 1]}'],
                'Predicted Positive': [f'FN: {cm_train[1, 0]}', f'TP: {cm_train[1, 1]}']
            },
            index=['Actual Negative', 'Actual Positive']
        )
        
        # Create formatted confusion matrices
        formatted_cm_test = pd.DataFrame(
            {
                'Predicted Negative': [f'TN: {cm_test[0, 0]}', f'FP: {cm_test[0, 1]}'],
                'Predicted Positive': [f'FN: {cm_test[1, 0]}', f'TP: {cm_test[1, 1]}']
            },
            index=['Actual Negative', 'Actual Positive']
        )
        
        # Assuming df is your DataFrame
        formatted_cm_train = formatted_cm_train.reset_index()  # Reset the index and make it a column
        formatted_cm_train.rename(columns={'index': 'Actual/Predicted'}, inplace=True)  # Rename the index column to "Measure"

        # Assuming df is your DataFrame
        formatted_cm_test = formatted_cm_test.reset_index()  # Reset the index and make it a column
        formatted_cm_test.rename(columns={'index': 'Actual/Predicted'}, inplace=True)  # Rename the index column to "Measure"

        # Display the formatted confusion matrices in Streamlit
        st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Confusion Matrix for Train (Development) Data</div>", unsafe_allow_html=True)
        # st.dataframe(formatted_cm_train, use_container_width=True)

        # df = df.drop(df.columns[0], axis=1)
        html_table = create_html_table_with_download(formatted_cm_train, "Count_Table.xlsx", image_path)
        st.markdown(html_table, unsafe_allow_html=True)

        # Function to plot confusion matrix using Plotly
        def plot_confusion_matrix(cm, title):
            # Create a figure for the confusion matrix with labeled positions
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale='viridis',
                colorbar=dict(title='Count'),
                hovertemplate='Count: %{z}<br>Predicted: %{x}<br>Actual: %{y}<extra></extra>'
            ))

            # Add text annotations for TN, FP, FN, TP
            annotations = [
                dict(x=0, y=0, text=f'TN: {cm[0, 0]}', showarrow=False, font=dict(color='white')),
                dict(x=1, y=0, text=f'FP: {cm[0, 1]}', showarrow=False, font=dict(color='white')),
                dict(x=0, y=1, text=f'FN: {cm[1, 0]}', showarrow=False, font=dict(color='white')),
                dict(x=1, y=1, text=f'TP: {cm[1, 1]}', showarrow=False, font=dict(color='white')),
            ]

            # Update layout with centered title and margins
            fig.update_layout(
                title=title,
                title_x=0.4,  # Center the title
                xaxis_title='Predicted',
                yaxis_title='Actual',
                width=600,
                height=400,
                margin=dict(l=1, r=10, t=90, b=50),  # Set margins
                annotations=annotations
            )
            return fig
        
        st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Confusion Matrix for Test (Monitoring) Data</div>", unsafe_allow_html=True)
        # st.dataframe(formatted_cm_test, use_container_width=True)

        html_table = create_html_table_with_download(formatted_cm_test, "Monitoring_Data.xlsx", image_path)
        st.markdown(html_table, unsafe_allow_html=True)
        
        ## Strength Statistics
        precision_train = precision_score(train_target, y_pred)
        precision_test= precision_score(test_target, y_pred_test)
        recall_train = recall_score(train_target, y_pred)
        recall_test = recall_score(test_target, y_pred_test)
        f1_train = f1_score(train_target, y_pred)
        f1_test = f1_score(test_target, y_pred_test)
        ks_statistic_train, p_value = stats.ks_2samp(train_target, y_pred)
        ks_statistic_test, p_value = stats.ks_2samp(test_target, y_pred_test)

        # Calculate Accuracy
        accuracy_train = accuracy_score(train_target, y_pred)
        accuracy_test = accuracy_score(test_target, y_pred_test)

        # Calculate Specificity
        specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1])  # TN / (TN + FP)
        specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])  # TN / (TN + FP)
        
        roc_auc = roc_auc_score(train_target, y_pred)
        gini_train =  2 * roc_auc - 1
        roc_auc_test = roc_auc_score(test_target, y_pred_test)
        gini_test=  2 * roc_auc_test - 1
        
        # Compile strength statistics
        strength_statistics = {
            'Performance Metrics': ['KS', 'Precision', 'F1-Score', 'Recall', 'Accuracy', 'Specificity', 'Gini', 'ROC AUC score'],
            'Development Data (Train Data)': [ks_statistic_train, precision_train, f1_train, recall_train, accuracy_train, specificity_train, gini_train, roc_auc],
            'Monitoring Data (Test Data)': [ks_statistic_test, precision_test, f1_test, recall_test, accuracy_test, specificity_test, gini_test, roc_auc_test]
        }
        strength_statistic = pd.DataFrame(strength_statistics).round(3)
        
        # wd.save_table(strength_statistic,'Strength Statistics')
        st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 16px;'>Strength Statistics</div>", unsafe_allow_html=True)
        # st.dataframe(strength_statistic, use_container_width=True)

        html_table = create_html_table_with_download(strength_statistic, "Strength_statistics.xlsx", image_path)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # # VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["Features"] = train_data.columns
        # calculating VIF for each feature
        vif_data["VIF"] = [np.round(variance_inflation_factor(train_data.values, i),4)
                                for i in range(len(train_data.columns))]
        
        # wd.save_table(np.round(vif_data,2),name="VIF Table")
        st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 16px;'>VIF Table</div>", unsafe_allow_html=True)
        # st.dataframe(vif_data, use_container_width=True)

        html_table = create_html_table_with_download(vif_data, "VIF_table.xlsx", image_path)
        st.markdown(html_table, unsafe_allow_html=True)
    
    
    with tab2:
        # Adding a small space between the title and the next section
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Graphs</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        

        # Function to plot ROC Curve with area highlighted
        def plot_roc_curve(fpr, tpr):
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
        
            # Highlight the area under the curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                fill='tozeroy',  # Highlight area under the curve
                mode='none',
                fillcolor='rgba(31, 119, 180, 0.5)',  # Color similar to your original code
                name=f'AUC = {roc_auc:.4f}'
            ))
        
            # Add ROC curve line
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                name='ROC Curve'
            ))
        
            # Add diagonal line
            fig.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=0, y1=1
            )
        
            fig.update_layout(
                title=f'ROC Curve (AUC={roc_auc:.4f})',
                title_x=0.2,  # Center the title
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600,
                height=450
            )
        
            return fig
        
        # Function to plot Precision-Recall Curve with area highlighted
        def plot_precision_recall_curve(precision, recall):
            pr_auc = auc(recall, precision)
            
            fig = go.Figure()
        
            # Highlight the area under the curve
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                fill='tozeroy',  # Highlight area under the curve
                mode='none',
                fillcolor='rgba(31, 119, 180, 0.5)',  # Color similar to your original code
                name=f'AUC = {pr_auc:.4f}'
            ))
        
            # Add Precision-Recall curve line
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                name='Precision-Recall Curve'
            ))
        
            # Add diagonal line
            fig.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=1, y1=0
            )
        
            fig.update_layout(
                title=f'Precision-Recall Curve (AUC={pr_auc:.4f})',
                title_x=0.2,  # Center the title
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=600,
                height=450
            )
        
            return fig
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Train Data</h5>", unsafe_allow_html=True)
            
            confusion_matrix_fig = plot_confusion_matrix(cm_train, "Confusion Matrix")
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)

            # Generate ROC curve data
            y_score = model.predict_proba(train_data)[:, 1]
            fpr, tpr, _ = roc_curve(train_target, y_score)
            
            # Plot ROC Curve
            roc_curve_fig = plot_roc_curve(fpr, tpr)
            st.plotly_chart(roc_curve_fig, use_container_width=True)
            
            # Generate Precision-Recall curve data
            precision, recall, _ = precision_recall_curve(train_target, y_score)
            
            # Plot Precision-Recall Curve
            precision_recall_fig = plot_precision_recall_curve(precision, recall)
            st.plotly_chart(precision_recall_fig, use_container_width=True)
            
        with col2:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Test Data</h5>", unsafe_allow_html=True)
            
            confusion_matrix_fig = plot_confusion_matrix(cm_test, "Confusion Matrix")
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)

            # Generate ROC curve data
            y_score_test = model.predict_proba(test_data)[:, 1]
            fpr, tpr, _ = roc_curve(test_target, y_score_test)
            
            # Plot ROC Curve
            roc_curve_fig = plot_roc_curve(fpr, tpr)
            st.plotly_chart(roc_curve_fig, use_container_width=True)
            
            # Generate Precision-Recall curve data
            precision, recall, _ = precision_recall_curve(test_target, y_score_test)
            
            # Plot Precision-Recall Curve
            precision_recall_fig = plot_precision_recall_curve(precision, recall)
            st.plotly_chart(precision_recall_fig, use_container_width=True)
    
if __name__ == "__main__":
    app()
