import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import html
from io import BytesIO
import base64
import pickle
from plotly.subplots import make_subplots
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve


# Streamlit app
def app():
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Stability Testing</h1>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(['Tables', 'Graphs'])
    with tab1:
        # Adding a small space between the title and the next section
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Ongoing Comparison</strong>
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
        
        col1 , col2 = st.columns(2)
        with col1:
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the CSV file
            csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
            
            # Read the CSV file
            df = pd.read_csv(csv_file_path_1)
            
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the pickle file
            model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
            
            # Load the pre-trained RandomForest model
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
            
            # Assuming the target variable is named 'target'
            target_column = 'def_flag'
            
            # Splitting the dataset into features (X) and target (y)
            X = df.drop(columns=[target_column])
            y = df[target_column]
        
            # Splitting the data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Assuming you have X_test and y_test as your test data
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            def calculate_csi(expected, actual, eps=1e-5):
                actual_perc = (actual / actual.sum()) + eps
                expected_perc = (expected / expected.sum()) + eps
                return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
            
            # Initialize a dictionary to store CSI scores for each feature
            csi_scores = {}
            
            # Calculate CSI for each feature except the target variable
            for feature in X.columns:
                expected = X_train[feature].value_counts().sort_index()
                actual = X_test[feature].value_counts().sort_index()
                csi_scores[feature] = calculate_csi(expected, actual)
            
            # Create a DataFrame to display the CSI scores
            csi_results = pd.DataFrame(list(csi_scores.items()), columns=["Features", "CSI Score"])

            csi_results = csi_results.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)
            
            st.markdown("<h5 style='text-align: center; font-size: 18px;'>Development Data</h5>", unsafe_allow_html=True)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; font-size: 15px;'>CSI Table</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(csi_results, use_container_width=True)

            # Base directory of the current script
            base_dir = os.path.dirname(__file__)

            # Construct the full path to the image file
            image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

            # df = df.drop(df.columns[0], axis=1)
            html_table = create_html_table_with_download(csi_results, "CSI_Table.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            def calculate_psi(expected, actual, bins=10, eps=1e-5):
                expected_perc, _ = np.histogram(expected, bins=bins)
                actual_perc, _ = np.histogram(actual, bins=bins)
                
                expected_perc = (expected_perc / len(expected)) + eps
                actual_perc = (actual_perc / len(actual)) + eps
                
                psi_value = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
                return psi_value
            
            # Example: Calculate PSI for `fico_t_bin`
            psi_fico_t_bin = calculate_psi(X_train['fico_t_bin'], X_test['fico_t_bin'])
            
            # Create a DataFrame to display the PSI result
            psi_results = pd.DataFrame({
                "Features": ["fico_t_bin"],
                "PSI Value": [psi_fico_t_bin]
            })
            
            psi_results = psi_results.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>PSI Table</div>", unsafe_allow_html=True)

            # # Display the DataFrame in Streamlit
            # st.dataframe(psi_results, use_container_width=True)
            
            html_table = create_html_table_with_download(psi_results, "psi_results.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            # ROC AUC Score
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # KS Statistic
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            ks_stat = max(tpr - fpr)
            
            # Precision
            precision = precision_score(y_test, y_pred)
            
            # F1-Score
            f1 = f1_score(y_test, y_pred)
            
            # Recall (Sensitivity)
            recall = recall_score(y_test, y_pred)
            
            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Specificity
            specificity = tn / (tn + fp)
            
            # Gini Coefficient
            gini = 2 * roc_auc - 1
            
            # Assuming the metrics are already calculated and stored in variables
            metrics = {
                "Performance Metrics": ["ROC AUC", "KS Statistic", "Precision", "F1-Score", "Recall (Sensitivity)", "Specificity", "Gini Coefficient"],
                "Development Data": [roc_auc, ks_stat, precision, f1, recall, specificity, gini]
            }
            
            # Creating the DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Strength Statistics</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(metrics_df, use_container_width=True) 

            metrics_df = metrics_df.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(metrics_df, "Strength_statistics.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True) 
            
            # Assuming the feature importance is already calculated and stored in variables
            feature_importance = pd.DataFrame({
                'Features': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Feature Importance</div>", unsafe_allow_html=True)
    
            # # Display the DataFrame in Streamlit
            # st.dataframe(feature_importance, use_container_width=True)

            feature_importance = feature_importance.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(feature_importance, "Feature_importance.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)

            # Calculate VIF for each feature
            vif_data = pd.DataFrame()
            vif_data['Features'] = X_test.columns
            vif_data['VIF'] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>VIF Table</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(vif_data, use_container_width=True)
            
            vif_data = vif_data.applymap(lambda x: "{:.5}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(vif_data, "vif_table.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
        
            def hosmer_lemeshow_test(y_true, y_pred_proba, bins=10):
                data = pd.DataFrame({
                    'true': y_true,
                    'prob': y_pred_proba
                })
                data['bin'] = pd.qcut(data['prob'], bins, duplicates='drop')
                grouped = data.groupby('bin')
            
                obs = grouped['true'].sum()
                exp = grouped['prob'].sum()
                obs_var = grouped['prob'].apply(lambda x: np.var(x))
            
                hl_stat = ((obs - exp) ** 2 / obs_var).sum()
                p_value = 1 - chi2.cdf(hl_stat, bins - 2)
            
                return hl_stat, p_value
            
            # Calculate the Hosmer-Lemeshow test statistic and p-value
            hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, y_pred_proba)
            
            # Format hl_stat to 4 decimal points
            hl_stat = round(hl_stat, 4)
            
            # Create a conclusion based on the test statistic and p-value
            conclusion = "The Model is not adequate" if hl_stat > hl_p_value else "The Model is adequate"
            
            # Create a DataFrame to display in Streamlit
            hl_test_results = pd.DataFrame({
                "Hosmer-Lemeshow Test": ["Test Statistic", "P-Value", "Conclusion"],
                "Value": [hl_stat, hl_p_value, conclusion]
            })
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Hosmer-Lemeshow Test</div>", unsafe_allow_html=True)
    
            # # Display the DataFrame in Streamlit
            # st.dataframe(hl_test_results, use_container_width=True)

            hl_test_results = hl_test_results.applymap(lambda x: "{:.10}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(hl_test_results, "HL_test.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
        with col2:
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the CSV file
            csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Monitoring_Data.csv')
            
            # Read the CSV file
            df1 = pd.read_csv(csv_file_path_2)
            
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the pickle file
            model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
            
            # Load the pre-trained RandomForest model
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
            
            # Assuming the target variable is named 'target'
            target_column = 'def_flag'
            
            # Splitting the dataset into features (X) and target (y)
            X = df1.drop(columns=[target_column])
            y = df1[target_column]
        
            # Splitting the data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Assuming you have X_test and y_test as your test data
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            def calculate_csi(expected, actual, eps=1e-5):
                actual_perc = (actual / actual.sum()) + eps
                expected_perc = (expected / expected.sum()) + eps
                return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
            
            # Initialize a dictionary to store CSI scores for each feature
            csi_scores = {}
            
            # Calculate CSI for each feature except the target variable
            for feature in X.columns:
                expected = X_train[feature].value_counts().sort_index()
                actual = X_test[feature].value_counts().sort_index()
                csi_scores[feature] = calculate_csi(expected, actual)
            
            # Create a DataFrame to display the CSI scores
            csi_results = pd.DataFrame(list(csi_scores.items()), columns=["Features", "CSI Score"])

            csi_results = csi_results.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)
            
            st.markdown("<h5 style='text-align: center; font-size: 18px;'>Monitoring Data</h5>", unsafe_allow_html=True)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; font-size: 15px;'>CSI Table</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(csi_results, use_container_width=True)

            # df = df.drop(df.columns[0], axis=1)
            html_table = create_html_table_with_download(csi_results, "CSI_Table.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            def calculate_psi(expected, actual, bins=10, eps=1e-5):
                expected_perc, _ = np.histogram(expected, bins=bins)
                actual_perc, _ = np.histogram(actual, bins=bins)
                
                expected_perc = (expected_perc / len(expected)) + eps
                actual_perc = (actual_perc / len(actual)) + eps
                
                psi_value = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
                return psi_value
            
            # Example: Calculate PSI for `fico_t_bin`
            psi_fico_t_bin = calculate_psi(X_train['fico_t_bin'], X_test['fico_t_bin'])
            
            # Create a DataFrame to display the PSI result
            psi_results = pd.DataFrame({
                "Features": ["fico_t_bin"],
                "PSI Value": [psi_fico_t_bin]
            })
            
            psi_results = psi_results.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>PSI Table</div>", unsafe_allow_html=True)

            # # Display the DataFrame in Streamlit
            # st.dataframe(psi_results, use_container_width=True)
            
            html_table = create_html_table_with_download(psi_results, "PSI_Table.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # ROC AUC Score
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # KS Statistic
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            ks_stat = max(tpr - fpr)
            
            # Precision
            precision = precision_score(y_test, y_pred)
            
            # F1-Score
            f1 = f1_score(y_test, y_pred)
            
            # Recall (Sensitivity)
            recall = recall_score(y_test, y_pred)
            
            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Specificity
            specificity = tn / (tn + fp)
            
            # Gini Coefficient
            gini = 2 * roc_auc - 1
            
            # Assuming the metrics are already calculated and stored in variables
            metrics = {
                "Performance Metrics": ["ROC AUC", "KS Statistic", "Precision", "F1-Score", "Recall (Sensitivity)", "Specificity", "Gini Coefficient"],
                "Development Data": [roc_auc, ks_stat, precision, f1, recall, specificity, gini]
            }
            
            # Creating the DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Strength Statistics</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(metrics_df, use_container_width=True) 

            metrics_df = metrics_df.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(metrics_df, "Strength_statistics.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)   
            
            # Assuming the feature importance is already calculated and stored in variables
            feature_importance = pd.DataFrame({
                'Features': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Feature Importance</div>", unsafe_allow_html=True)
    
            # # Display the DataFrame in Streamlit
            # st.dataframe(feature_importance, use_container_width=True)

            feature_importance = feature_importance.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(feature_importance, "Feature_importance.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Calculate VIF for each feature
            vif_data = pd.DataFrame()
            vif_data['Features'] = X_test.columns
            vif_data['VIF'] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>VIF Table</div>", unsafe_allow_html=True)
            
            # # Display the DataFrame in Streamlit
            # st.dataframe(vif_data, use_container_width=True)
            
            vif_data = vif_data.applymap(lambda x: "{:.5}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(vif_data, "VIF_Table.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            def hosmer_lemeshow_test(y_true, y_pred_proba, bins=10):
                data = pd.DataFrame({
                    'true': y_true,
                    'prob': y_pred_proba
                })
                data['bin'] = pd.qcut(data['prob'], bins, duplicates='drop')
                grouped = data.groupby('bin')
            
                obs = grouped['true'].sum()
                exp = grouped['prob'].sum()
                obs_var = grouped['prob'].apply(lambda x: np.var(x))
            
                hl_stat = ((obs - exp) ** 2 / obs_var).sum()
                p_value = 1 - chi2.cdf(hl_stat, bins - 2)
            
                return hl_stat, p_value
            
            # Calculate the Hosmer-Lemeshow test statistic and p-value
            hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, y_pred_proba)
            
            # Format hl_stat to 4 decimal points
            hl_stat = round(hl_stat, 4)
            
            # Create a conclusion based on the test statistic and p-value
            conclusion = "The Model is not adequate" if hl_stat > hl_p_value else "The Model is adequate"
            
            # Create a DataFrame to display in Streamlit
            hl_test_results = pd.DataFrame({
                "Hosmer-Lemeshow Test": ["Test Statistic", "P-Value", "Conclusion"],
                "Value": [hl_stat, hl_p_value, conclusion]
            })
            
            # Add title in the middle of the table using st.markdown
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 15px;'>Hosmer-Lemeshow Test</div>", unsafe_allow_html=True)
    
            # # Display the DataFrame in Streamlit
            # st.dataframe(hl_test_results, use_container_width=True)

            hl_test_results = hl_test_results.applymap(lambda x: "{:.10}".format(x) if isinstance(x, (int, float)) else x)

            html_table = create_html_table_with_download(hl_test_results, "HL_test.xlsx", image_path)
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
        
        # Base directory of the current script
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the CSV file
        csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path_1)
        
        # Base directory of the current script
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the pickle file
        model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
        
        # Load the pre-trained RandomForest model
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
        
        # Assuming the target variable is named 'target'
        target_column = 'def_flag'
        
        # Splitting the dataset into features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
        # Splitting the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Assuming you have X_test and y_test as your test data
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Function to plot confusion matrix
        def plot_confusion_matrix(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Positive', 'Negative'],
                y=['Positive', 'Negative'],
                colorscale='Viridis',
                hoverinfo='z',
                text=cm,  # Add text values to the heatmap
                texttemplate='%{text}',  # Format the text to display values
                colorbar=dict(title='Counts')
            ))
        
            # Update layout with your desired styling
            fig.update_layout(
                title='Confusion Matrix',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
                xaxis_title='Predicted Values',
                yaxis_title='Actual Values',
                legend=dict(
                    orientation='h',
                    x=0.2,
                    xanchor='left',
                    y=-0.30,
                    traceorder='normal',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                ),
                margin=dict(l=10, r=10, t=90, b=80),
                width=600,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white',
                coloraxis_colorbar=dict(title='Counts')  # Adding a title for the color bar
            )
        
            return fig
        
        def plot_roc_curve(y_true, y_proba):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            
            fig = go.Figure()
        
            # Add the ROC curve
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#003366')))  # Customize line color
        
            # Add the random guessing line
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(dash='dash', color='red')))
        
            # Update layout with desired styling
            fig.update_layout(
                title=f'ROC Curve (AUC = {roc_auc:.2f})',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(
                    orientation='h',
                    x=0.2,
                    xanchor='left',
                    y=-0.30,
                    traceorder='normal',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                ),
                margin=dict(l=10, r=10, t=90, b=80),
                width=600,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
        
        # Function to plot KS statistics with specified formatting
        def plot_ks_statistics(y_true, y_proba):
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            ks_stat = max(tpr - fpr)
        
            fig = go.Figure()
        
            # Add True Positive Rate trace
            fig.add_trace(go.Scatter(x=thresholds, y=tpr, mode='lines', name='True Positive Rate', line=dict(color='#003366')))  # Customize line color
        
            # Add False Positive Rate trace
            fig.add_trace(go.Scatter(x=thresholds, y=fpr, mode='lines', name='False Positive Rate', line=dict(color='#FF6692')))  # Customize line color
        
            # Update layout with desired styling
            fig.update_layout(
                title=f'KS Statistic = {ks_stat:.2f}',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
                xaxis_title='Threshold',
                yaxis_title='Percentage Below Threshold',
                legend=dict(
                    orientation='h',
                    x=0.2,
                    xanchor='left',
                    y=-0.30,
                    traceorder='normal',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                ),
                margin=dict(l=10, r=10, t=90, b=80),
                width=600,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        
            return fig
        
        
        # Function to plot Sensitivity and Specificity on a 3-axis plot
        def plot_sensitivity_specificity_cutoff(y_true, y_proba):
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            specificity = 1 - fpr
        
            # Create subplots with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        
            # Add Sensitivity trace (on primary y-axis)
            fig.add_trace(go.Scatter(x=thresholds, y=tpr, mode='lines', name='Sensitivity', line=dict(color='#FF6692')), secondary_y=False)
        
            # Add Specificity trace (on secondary y-axis)
            fig.add_trace(go.Scatter(x=thresholds, y=specificity, mode='lines', name='Specificity', line=dict(color='#003366')), secondary_y=True)
        
            # Update layout
            fig.update_layout(
                title='Sensitivity VS Specificity',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
                xaxis_title='Cutoff',
                yaxis_title='Sensitivity',
                legend=dict(
                    orientation='h',
                    x=0.2,
                    xanchor='left',
                    y=-0.30,
                    traceorder='normal',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                ),
                margin=dict(l=10, r=10, t=90, b=80),
                width=600,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        
            # Update y-axis titles
            fig.update_yaxes(title_text='Sensitivity', secondary_y=False)
            fig.update_yaxes(title_text='Specificity', secondary_y=True)
        
            return fig
        
        # Function to plot Feature Importance with features on the x-axis
        def plot_feature_importance(feature_importance):
            fig = go.Figure(data=[go.Bar(
                x=feature_importance['Features'],  # Features on the x-axis
                y=feature_importance['Importance'],  # Importance on the y-axis
                marker=dict(color='#003366'),  # Custom color for the bars
            )])
            
            # Update layout
            fig.update_layout(
                title='Feature Importance',
                title_x=0.3,  # Center the title
                title_font=dict(size=17, color='black', family='Calibri'),
                xaxis_title='Features',
                yaxis_title='Feature Importance',
                legend=dict(
                    orientation='h',
                    x=0.2,
                    xanchor='left',
                    y=-0.30,
                    traceorder='normal',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                ),
                margin=dict(l=10, r=10, t=90, b=80),
                width=600,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        
            return fig
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Development Data</h5>", unsafe_allow_html=True)
            # Example usage
            confusion_matrix_fig = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
            
            # Example usage
            roc_curve_fig = plot_roc_curve(y_test, y_pred_proba)
            st.plotly_chart(roc_curve_fig, use_container_width=True)
            
            # Example usage
            ks_statistics_fig = plot_ks_statistics(y_test, y_pred_proba)
            st.plotly_chart(ks_statistics_fig, use_container_width=True)
            
            # Example usage
            sensitivity_specificity_fig = plot_sensitivity_specificity_cutoff(y_test, y_pred_proba)
            st.plotly_chart(sensitivity_specificity_fig, use_container_width=True)
            
            # Example usage
            feature_importance_fig = plot_feature_importance(feature_importance)
            st.plotly_chart(feature_importance_fig, use_container_width=True)
            
        with col2:
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the CSV file
            csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Monitoring_Data.csv')
            
            # Read the CSV file
            df1 = pd.read_csv(csv_file_path_2)
            
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the pickle file
            model_file_path = os.path.join(base_dir, 'pkl', 'Random_forest.pkl')
            
            # Load the pre-trained RandomForest model
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
            
            # Assuming the target variable is named 'target'
            target_column = 'def_flag'
            
            # Splitting the dataset into features (X) and target (y)
            X = df1.drop(columns=[target_column])
            y = df1[target_column]
        
            # Splitting the data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Assuming you have X_test and y_test as your test data
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Monitoring Data</h5>", unsafe_allow_html=True)
            # Example usage
            confusion_matrix_fig = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
            
            # Example usage
            roc_curve_fig = plot_roc_curve(y_test, y_pred_proba)
            st.plotly_chart(roc_curve_fig, use_container_width=True)
            
            # Example usage
            ks_statistics_fig = plot_ks_statistics(y_test, y_pred_proba)
            st.plotly_chart(ks_statistics_fig, use_container_width=True)
            
            # Example usage
            sensitivity_specificity_fig = plot_sensitivity_specificity_cutoff(y_test, y_pred_proba)
            st.plotly_chart(sensitivity_specificity_fig, use_container_width=True)
            
            # Example usage
            feature_importance_fig = plot_feature_importance(feature_importance)
            st.plotly_chart(feature_importance_fig, use_container_width=True)
    

if __name__ == "__main__":
    app()
