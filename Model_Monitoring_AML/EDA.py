import streamlit as st
import pandas as pd
import os
import html
from io import BytesIO
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Streamlit app
def app():
    # Centered text in the floating container
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Exploratory Data Analysis</h1>
        """,
        unsafe_allow_html=True
    )
    
    tab1, tab2, tab3 = st.tabs(['Graphs', 'Data Quality Check', 'Summary Statistics'])
    
    with tab1:
        # Adding a small space between the title and the next section
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Graphical Representation</strong>
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
        
        # Bar chart using go.Figure instead of px
        def plot_total_transactions_vs_risk(df):
            fig = go.Figure()

            # Add bar chart trace
            fig.add_trace(go.Bar(
                x=df['CUSTOMER_RISK_LEVEL'],
                y=df[' TOT_TRANS'],
                marker_color='rgb(31, 119, 180)',  # You can customize colors here
                name='Total Transactions'
            ))

            # Update layout with centered title, axis labels, and margins
            fig.update_layout(
                title='Total Transactions vs Customer Risk Level',
                title_x=0.1,  # Center the title
                xaxis_title='Customer Risk Level',
                yaxis_title='Total Transactions',
                width=600,
                height=400,
                margin=dict(l=10, r=10, t=90, b=80)  # Set left, right, top, and bottom margins
            )

            return fig
        
    
        # Pie chart using go.Figure instead of px
        def plot_customer_risk_distribution(df):
            fig = go.Figure()

            # Add pie chart trace
            fig.add_trace(go.Pie(
                labels=df['CUSTOMER_RISK_LEVEL'],  # Categories for pie chart
                values=df['CUSTOMER_RISK_LEVEL'].value_counts(),  # Corresponding counts for each category
                name='Customer Risk Levels',
                hole=0.3  # Optional: make it a donut chart by setting the hole
            ))

            # Update layout with centered title and margins
            fig.update_layout(
                title='Distribution of Customer Risk Levels',
                title_x=0.1,  # Center the title
                width=600,
                height=400,
                margin=dict(l=10, r=10, t=50, b=50)  # Set left, right, top, and bottom margins
            )

            return fig

        
        # Function to plot histogram of Total Transactions
        def plot_total_transactions_histogram(df, nbins=10):
            fig = go.Figure()

            # Add histogram trace
            fig.add_trace(go.Histogram(
                x=df[' TOT_TRANS'],  # Data for the histogram
                nbinsx=nbins,  # Number of bins
                name='Total Transactions',
                marker_color='rgba(31, 119, 180, 0.9)'  # Optional: set a specific color
            ))

            # Update layout with centered title and margins
            fig.update_layout(
                title='Histogram of Total Transactions',
                title_x=0.25,  # Center the title
                xaxis_title='Total Transactions',
                yaxis_title='Count',
                width=600,
                height=400,
                margin=dict(l=10, r=10, t=90, b=50)  # Set left, right, top, and bottom margins
            )

            return fig

        
        # Function to plot a box plot of Total Transactions by Customer Risk Level
        def plot_total_transactions_boxplot(df):
            fig = go.Figure()

            # Add box plot trace
            fig.add_trace(go.Box(
                x=df['CUSTOMER_RISK_LEVEL'],  # Category (Customer Risk Level) on the x-axis
                y=df[' TOT_TRANS'],  # Total Transactions on the y-axis
                name='Total Transactions',
                marker_color='rgba(31, 119, 180, 0.9)'  # Optional: set a specific color
            ))

            # Update layout with centered title and margins
            fig.update_layout(
                title='Distribution of Total Transactions by Customer Risk Level',
                title_x=0,  # Center the title
                xaxis_title='Customer Risk Level',
                yaxis_title='Total Transactions',
                width=600,
                height=400,
                margin=dict(l=1, r=10, t=90, b=50)  # Set left, right, top, and bottom margins
            )

            return fig


        # Function to plot a correlation heatmap
        def plot_correlation_heatmap(df):
            corr = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                colorbar=dict(title='Correlation'),
                text=corr.values,  # Add text annotations
                texttemplate='%{text:.2f}',  # Format the text to 2 decimal places
            ))

            # Update layout with centered title and margins
            fig.update_layout(
                title='Correlation Heatmap',
                title_x=0.3,  # Center the title
                width=600,
                height=400,
                margin=dict(l=1, r=10, t=90, b=50),  # Set left, right, top, and bottom margins
                xaxis=dict(tickfont=dict(size=9)),  # Adjust font size for x-axis
                yaxis=dict(tickfont=dict(size=9))   # Adjust font size for y-axis
            )

            return fig
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Development Data</h5>", unsafe_allow_html=True)

            # Generate and show all figures
            total_transactions_fig = plot_total_transactions_vs_risk(df)
            st.plotly_chart(total_transactions_fig, use_container_width=True)
            
            pie_chart_fig = plot_customer_risk_distribution(df)
            st.plotly_chart(pie_chart_fig, use_container_width=True)
                
            histogram_fig = plot_total_transactions_histogram(df, nbins=10)
            st.plotly_chart(histogram_fig, use_container_width=True)
                
            boxplot_fig = plot_total_transactions_boxplot(df)
            st.plotly_chart(boxplot_fig, use_container_width=True)
                
            heatmap_fig = plot_correlation_heatmap(df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            
        with col2:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Monitoring Data</h5>", unsafe_allow_html=True)

            # Generate and show all figures
            total_transactions_fig = plot_total_transactions_vs_risk(df1)
            st.plotly_chart(total_transactions_fig, use_container_width=True)
            
            pie_chart_fig = plot_customer_risk_distribution(df1)
            st.plotly_chart(pie_chart_fig, use_container_width=True)
                
            histogram_fig = plot_total_transactions_histogram(df1, nbins=10)
            st.plotly_chart(histogram_fig, use_container_width=True)
                
            boxplot_fig = plot_total_transactions_boxplot(df1)
            st.plotly_chart(boxplot_fig, use_container_width=True)
                
            heatmap_fig = plot_correlation_heatmap(df1)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
    with tab2:
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Box Plot</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the CSV file
            csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
            
            # Read the CSV file
            df = pd.read_csv(csv_file_path_1)
            
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Development Data</h5>", unsafe_allow_html=True)
            
            # Dropdown for column selection
            selected_columns = st.multiselect("Select Columns for Box Plot", options=df.columns.tolist(), key="dev_columns")

            if selected_columns:
                # Create a box plot for the selected columns
                fig = make_subplots(rows=1, cols=len(selected_columns))
            
                for i, var in enumerate(selected_columns):
                    fig.add_trace(go.Box(y=df[var], name=var), row=1, col=i + 1)
            
                fig.update_traces(boxpoints='all', jitter=.3)
                fig.update_layout(title_text="Box Plot", width=600, height=400)
            
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one column to display the box plot.")
                
        with col2:
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the CSV file
            csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Monitoring_Data.csv')
            
            # Read the CSV file
            df1 = pd.read_csv(csv_file_path_2)
            
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Monitoring Data</h5>", unsafe_allow_html=True)
            
            # Dropdown for column selection
            selected_columns = st.multiselect("Select Columns for Box Plot", options=df1.columns.tolist(), key="mon_columns")

            if selected_columns:
                # Create a box plot for the selected columns
                fig = make_subplots(rows=1, cols=len(selected_columns))
            
                for i, var in enumerate(selected_columns):
                    fig.add_trace(go.Box(y=df1[var], name=var), row=1, col=i + 1)
            
                fig.update_traces(boxpoints='all', jitter=.3)
                fig.update_layout(title_text="Box Plot", width=600, height=400)
            
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one column to display the box plot.")
        
        with tab3:
            st.markdown(
                """
                <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                    <strong>Summary Statistics</strong>
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
            
            col1, col2 = st.columns(2)
            with col1:
                # Base directory of the current script
                base_dir = os.path.dirname(__file__)
                
                # Construct the full path to the CSV file
                csv_file_path_1 = os.path.join(base_dir, 'Datasets', 'Development_Data.csv')
                
                # Read the CSV file
                df = pd.read_csv(csv_file_path_1)
                
                result=df.describe()
                
                st.markdown("<h5 style='text-align: center; font-size: 18px;'>Development Data</h5>", unsafe_allow_html=True)
    
                # st.dataframe(result, use_container_width=True)
                # Base directory of the current script
                base_dir = os.path.dirname(__file__)

                # Construct the full path to the image file
                image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

                # df = df.drop(df.columns[0], axis=1)
                html_table = create_html_table_with_download(result, "Development_Summary.xlsx", image_path)
                st.markdown(html_table, unsafe_allow_html=True)
            
            with col2:
                # Base directory of the current script
                base_dir = os.path.dirname(__file__)
                
                # Construct the full path to the CSV file
                csv_file_path_2 = os.path.join(base_dir, 'Datasets', 'Monitoring_Data.csv')
                
                # Read the CSV file
                df1 = pd.read_csv(csv_file_path_2)
                
                result1=df1.describe()
                
                st.markdown("<h5 style='text-align: center; font-size: 18px;'>Monitoring Data</h5>", unsafe_allow_html=True)
    
                # st.dataframe(result1, use_container_width=True)
                html_table = create_html_table_with_download(result, "Monitoring_Summary.xlsx", image_path)
                st.markdown(html_table, unsafe_allow_html=True)

   
if __name__ == "__main__":
    app()
