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
        
        # Function to create scatter plot for CLTV vs FICO
        def scatter_plot_cltv_vs_fico(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['fico_t_bin'], 
                y=df['cltv'], 
                mode='markers', 
                name='CLTV vs FICO',  # Legend name
                marker=dict(color=df['def_flag'], colorscale='Viridis', size=10),
                text=df['def_flag'],  # Tooltip showing default flag
            ))
            fig.update_layout(
                title='CLTV vs FICO',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
                xaxis_title='FICO Score',
                yaxis_title='CLTV'
            )
            return fig
        
        # Function to create scatter plot for DTI vs Interest Rate
        def scatter_plot_dti_vs_int_rt(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['dti'], 
                y=df['int_rt'], 
                mode='markers', 
                name='DTI vs Interest Rate',  # Legend name
                marker=dict(color=df['def_flag'], colorscale='Viridis', size=10),
                text=df['def_flag'],
            ))
            fig.update_layout(
                title='DTI vs Interest Rate',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
                xaxis_title='DTI',
                yaxis_title='Interest Rate'
            )
            return fig
        
        # Function to create box plot for FICO by Default Flag
        def box_plot_fico_by_def_flag(df):
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df['fico_t_bin'],
                x=df['def_flag'], 
                boxmean='sd' , # Show mean and standard deviation
                name='FICO by Default Flag',  # Legend name
                marker=dict(color='#AB63FA'),
            ))
            fig.update_layout(
                title='FICO Score by Default Flag',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
                xaxis_title='Default Flag',
                yaxis_title='FICO Score'
            )
            return fig
        
        # Function to create histogram for CLTV
        def histogram_cltv(df):
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['cltv'],
                nbinsx=20,
                name='CLTV Distribution',  # Legend name
                marker=dict(color='#003366'),
            ))
            fig.update_layout(
                title='Distribution of CLTV',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
                xaxis_title='CLTV',
                yaxis_title='Count'
            )
            return fig
        
        # Function to create bar graph for ECL Amount by Stage
        def bar_graph_cltv_by_stage(df):
            ecl_by_stage = df.groupby('def_flag')['cltv'].sum().reset_index()  # Replace 'cltv' with your ECL Amount column
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ecl_by_stage['def_flag'],
                y=ecl_by_stage['cltv'],
                name='CLTV Amount by Default Flag',  # Legend name
                marker=dict(color='#003366'),
            ))
            fig.update_layout(
                title='CLTV Amount by Default Flag',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
                xaxis_title='Default Flag',
                yaxis_title='CLTV Amount'
            )
            return fig
        
        # Function to create heatmap for correlation matrix
        def heatmap_correlation(df):
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
            fig.update_layout(
                title='Correlation Matrix Heatmap',
                title_x=0.3,
                title_font=dict(size=17, color='black', family='Calibri'),
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
            # Generate and show all figures
            fig1 = scatter_plot_cltv_vs_fico(df)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = scatter_plot_dti_vs_int_rt(df)
            st.plotly_chart(fig2, use_container_width=True)
            
            fig3 = box_plot_fico_by_def_flag(df)
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = histogram_cltv(df)
            st.plotly_chart(fig4, use_container_width=True)
            
            fig5 = bar_graph_cltv_by_stage(df)
            st.plotly_chart(fig5, use_container_width=True)
            
            fig6 = heatmap_correlation(df)
            st.plotly_chart(fig6, use_container_width=True)
            
        with col2:
            st.markdown("<h5 style='text-align: center; font-size: 17px;'>Monitoring Data</h5>", unsafe_allow_html=True)
            # Generate and show all figures
            fig7 = scatter_plot_cltv_vs_fico(df1)
            st.plotly_chart(fig7, use_container_width=True)
            
            fig8 = scatter_plot_dti_vs_int_rt(df1)
            st.plotly_chart(fig8, use_container_width=True)
            
            fig9 = box_plot_fico_by_def_flag(df1)
            st.plotly_chart(fig9, use_container_width=True)
            
            fig10 = histogram_cltv(df1)
            st.plotly_chart(fig10, use_container_width=True)
            
            fig11 = bar_graph_cltv_by_stage(df1)
            st.plotly_chart(fig11, use_container_width=True)
            
            fig12 = heatmap_correlation(df1)
            st.plotly_chart(fig12, use_container_width=True)
            
    with tab2:
        st.markdown(
            """
            <div style='margin-top: 1px; margin-bottom: 7px; text-align: center; font-size: 20px;'>
                <strong>Box Plot</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # # Inject custom CSS to change the border color of input box and multiselect dropdown
        # st.markdown("""
        #     <style>
        #     input[type=number] {
        #         border: 1px solid rgb(39, 45, 85);
        #         border-radius: 8px;
        #         padding: 5px;
        #     }
        #     div[data-baseweb="select"] > div {
        #         border: 1px solid rgb(39, 45, 85) !important;
        #         border-radius: 8px;
        #     }
        #     </style>
        #     """, unsafe_allow_html=True)

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
                
                result = df.describe()

                result = result.applymap(lambda x: "{:.5}".format(x) if isinstance(x, (int, float)) else x)
                
                st.markdown("<h5 style='text-align: center; font-size: 18px;'>Development Data</h5>", unsafe_allow_html=True)

                # Assuming df is your DataFrame
                result = result.reset_index()  # Reset the index and make it a column
                result.rename(columns={'index': 'Measures'}, inplace=True)  # Rename the index column to "Measure"
    
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
                
                result =df1.describe()
                
                result = result.applymap(lambda x: "{:.4}".format(x) if isinstance(x, (int, float)) else x)

                st.markdown("<h5 style='text-align: center; font-size: 18px;'>Monitoring Data</h5>", unsafe_allow_html=True)

                # Assuming df is your DataFrame
                result = result.reset_index()  # Reset the index and make it a column
                result.rename(columns={'index': 'Measures'}, inplace=True)  # Rename the index column to "Measure"

                # df = df.drop(df.columns[0], axis=1)
                html_table = create_html_table_with_download(result, "Monitoring_Summary.xlsx", image_path)
                st.markdown(html_table, unsafe_allow_html=True)
    
                # st.dataframe(result1, use_container_width=True)

   
if __name__ == "__main__":
    app()
