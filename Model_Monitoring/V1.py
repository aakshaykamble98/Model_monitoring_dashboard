import streamlit as st
from PIL import Image
import pandas as pd
import os
import html
import plotly.graph_objs as go
from io import BytesIO
import base64
from streamlit_option_menu import option_menu
import EDA, Stability_Testing, Model_Monitoring, Back_Testing

# Construct the path to the image
image_path = os.path.join(os.path.dirname(__file__), 'Images', 'NIMBUS_logo.png')
image = Image.open(image_path)

#Setting wide layout for streamlit app
st.set_page_config(
    page_title="Model Monitoring Dashboard", layout="wide", page_icon=image
)

# #CSS to hide the Streamlit stop button
# hide_st_button = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stApp > header {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_st_button, unsafe_allow_html=True)

# Base directory of the current script
base_dir = os.path.dirname(__file__)

# Construct the full path to the image file
image_path = os.path.join(base_dir, 'Images', 'NIMBUS_Uno.png')

# Convert the image to base64 encoding
with open(image_path, "rb") as img_file:
    encoded_image = base64.b64encode(img_file.read()).decode()

# Merged CSS for header, sidebar, and additional elements
st.markdown(f"""
<style>
    /* Custom header styling with embedded base64 image */
    header[data-testid="stHeader"] {{
        background-color: rgb(39, 45, 85); /* Dark blue background */
        height: 55px; /* Adjust height of the header */
        display: flex;
        align-items: center;
        justify-content: flex-start; /* Align content to the left */
        padding-left: 10px; /* Adjust padding for logo spacing */
        color: white;
    }}

    /* Logo styling */
    header[data-testid="stHeader"]::before {{
        content: '';
        background-image: url('data:image/png;base64,{encoded_image}'); /* Use base64 image */
        background-repeat: no-repeat;
        background-size: contain;
        display: block;
        height: 40px; /* Adjust height of the logo */
        width: 120px; /* Adjust width of the logo */
        margin-left: 25px; /* Space between the logo and the header content */
    }}

    /* Adjustments for emotion cache and layout spacing */
    div[class^='st-emotion-cache-10oheav'] {{
        padding-top: 0rem;
    }}

    div[class^='css-1544g2n'] {{
        padding-top: 0rem;
    }}

    /* Radio button styling */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {{
        background-color: #b5d6fd;
        padding-right: 10px;
        padding-left: 4px;
        padding-bottom: 3px;
        margin: 4px;
    }}

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {{
        top: 55px; 
        background-color: #84AFEF33;
        padding-top: 0px;
    }}

    section[data-testid="stSidebar"] h1 {{
        color: white;
    }}

    section[data-testid="stSidebar"] label {{
        color: black;
    }}

    /* Report view container background */
    .reportview-container {{
        background: white;
    }}

    /* Title styling */
    .title {{
        color: black;
        font-size: 25px;
        text-align: left;
        margin: 10px 0;
    }}

    /* Center align class */
    .center {{
        text-align: center;
    }}

    /* Sidebar header gradient */
    section[data-testid="stSidebar"] div[class="css-17eq0hr e1fqkh3o1"] {{
        background-image: linear-gradient(#8993ab,#8993ab);
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Inject custom CSS to remove space from the top of the page
custom_css = """
    /* Remove top padding from the main block of the Streamlit app */
    .block-container {
        padding-top: 2.5rem !important;
    }

    /* Optional: Remove margin around the header */
    header, .stApp {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Change color of the sidebar toggle button */
    [data-testid="collapsedControl"] {
        color: white !important;
    }

    /* Change the background color of the main page */
    .stApp {
        background-color: #f2f9f9;  /* Change this to your desired color */
    }
"""

# Apply the custom CSS to your Streamlit app
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


# Inject custom CSS to change the border color of input box and multiselect dropdown
st.markdown("""
    <style>
    input[type=number] {
        border: 1px solid rgb(39, 45, 85);
        border-radius: 8px;
        padding: 5px;
    }
    div[data-baseweb="select"] > div {
        border: 1px solid rgb(39, 45, 85) !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Model Monitoring Dashboard',
                options=['Overview', 'Data Overview', 'EDA', 'Stability Testing', 'Model Monitoring', 'Back Testing'],
                icons=['cast', 'book', 'bar-chart', 'shield-check', 'graph-up', 'clock-history'],
                menu_icon='bank',
                default_index=0,
                styles={
                "container": {"padding": "0px", "background-color": "rgb(132, 175, 239, 0.2)", "border": "1.1px solid #A9A9A9", "border-radius": "7px",},
                "icon": {"color": "#272d55", "font-size": "17px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "color": "#272d55", "--hover-color": "#c1d6fd", "font-family": "sans-serif"},
                "nav-link-selected": {"background-color": "#AFDBF5", "border-left": "3px solid #272d55","color": "#272d55",},
                "menu-title": {"text-align": "center", "font-weight": "bold", "font-size": "20px", "margin": "0px"},
            }
            )


        if app == "Overview":
            # Centered text in the floating container
            st.markdown(
                """
                <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Welcome to Model Monitoring Dashboard</h1>
                """,
                unsafe_allow_html=True
            )
            
            # # Define custom CSS to center text and set font size
            # custom_css = """
            # <style>
            # .centered-overview {
            #     text-align: center;
            #     font-size: 22px;
            # }
            # </style>
            # """
            # st.markdown(custom_css, unsafe_allow_html=True) 
            
            # # Use the custom CSS class to center and resize the text
            # st.markdown("<div class='centered-overview'><b>Overview</b></div>", unsafe_allow_html=True)
            
            st.markdown(
                    """
                    <div style="text-align: justify;">
                    <p>The model monitoring dashboard provides essential insights into the stability index of our machine learning models, allowing users to evaluate their performance over time. It enables the generation of new models for comparison and assessment against baseline metrics. This intuitive interface ensures optimal model performance and facilitates timely adjustments based on evolving data conditions.</p>
                    </div>
                    """,

                    unsafe_allow_html=True
                )
            
            # Base directory of the current script
            base_dir = os.path.dirname(__file__)
            
            # Construct the full path to the image file
            image_path = os.path.join(base_dir, 'Images', 'Overview_1.png')
            
            # Display the image below the text
            st.image(image_path, use_column_width=True)
            

        elif app == "Data Overview":
            st.markdown(
                """
                <h1 style='text-align: center; font-size: 28px; color: rgb(39, 45, 85);'>Data Overview</h1>
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
                    <strong>Development Data</strong>
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
            # df['int_rt'] = df['int_rt'].apply(lambda x: "{:.3}".format(x))
            df['fico_t_bin'] = df['fico_t_bin'].apply(lambda x: "{:.3}".format(x)) 
            df1['fico_t_bin'] = df1['fico_t_bin'].apply(lambda x: "{:.3}".format(x))            
            
            # st.dataframe(df, use_container_width=True)

            # Base directory of the current script
            base_dir = os.path.dirname(__file__)

            # Construct the full path to the image file
            image_path = os.path.join(base_dir, 'Images', 'Download_icon.png')

            # df = df.drop(df.columns[0], axis=1)
            html_table = create_html_table_with_download(df, "Development_Data.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Adding a small space between the title and the next section
            st.markdown(
                """
                <div style='margin-top: 15px; margin-bottom: 2px; text-align: center; font-size: 20px;'>
                    <strong>Monitoring Data</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # st.dataframe(df1, use_container_width=True)

            # df1 = df1.drop(df1.columns[0], axis=1)
            html_table = create_html_table_with_download(df1, "Monitoring_Data.xlsx", image_path)
            st.markdown(html_table, unsafe_allow_html=True)
            
        else:
            selected_app = next((a for a in self.apps if a["title"] == app), None)
            if selected_app:
                selected_app["function"].app()


if __name__ == "__main__":
    multi_app = MultiApp()
    
    multi_app.add_app("EDA", EDA)
    multi_app.add_app("Stability Testing", Stability_Testing)
    multi_app.add_app("Model Monitoring", Model_Monitoring)
    multi_app.add_app("Back Testing", Back_Testing)
    
    multi_app.run()
