#STREAMLIT APP


import streamlit as st
import pandas as pd
import numpy as np
# Set the page icon and layout type
st.set_page_config(page_title="CityZen App", page_icon="location.png", layout="wide")

#streamlit run [app.py]
from numpy import array
# Example: Importing specific data types from NumPy
from numpy import int32, float64
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
import adjustText
#import wikipedia
#import wikipediaapi

# Load the data
import geopandas as gpd
import topojson as tp
import shapely
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon

import time
import base64
from pages.generate_recommendations import generate_recommendations
from pages.about import about_page
#from app import app

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

#st.success('This is a success message!', icon="✅")


# Initialize Wikipedia API
#wiki_api = wikipediaapi.Wikipedia('en')

geometry = gpd.read_file("geometry_19.topojson")

landkreise_scaled = pd.read_csv("landkreise_scaled.csv")

# Define the list of 20 features
all_features = landkreise_scaled.columns.to_list()[1:]

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        
        background-size: cover
        
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('cityzen_22.jpg')  

# Add CSS for button styling
st.markdown(
    """
    <style>
    .unclicked-button {
        background-color: #ffffff;
        color: #000000;
    }
    .clicked-button {
        background-color: #000000;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#@st.cache_data(experimental_allow_widgets=True)
# Create a Streamlit app
def main():
    
    # Set the app title
    st.sidebar.title('CityZen')
    st.sidebar.header("Find your Zen City")

    st.sidebar.header("Which characteristics of cities are most important to you?")

    # Select 6 prioritized features
    #st.header('Select 6 Prioritized Features')
    # Create an empty placeholder to hold the selected features
    selected_features = []
    selected_features_placeholder = st.empty()
    submit_button = None
        # Sort the selected features in alphabetical order
    #selected_features.sort()
    # Initialize selected features list in session state
    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = []

    

    # Define the categories and buttons within each category
    categories = {
        'Affordability': {
        'icon': 'funding.png',
        'buttons': [ 'low unemployment',
    'low land prices',                                                
    'low rental prices',                                                    
    'high income',                                              
    'high purchasing power', 'more shopping malls', 'more business places']},
            'Social': {
        'icon': 'social.png',
        'buttons': [ 'Child care friendly', 'low avg age', 'high avg age',
    'high cultural diversity', 'high tourism', 'low tourism', 'more restaurants',  'high capacity of hospitals',                                            
    'more associations',  'more sports and leisure activities']},
            'Environmental': {
        'icon': 'city.png',
        'buttons':[ 'low industry', 'high industry',                                            
    'low rurality', 'high rurality', 'low population density',                                              
       'high coastal area',                                                 
    'high green spaces',                                 
    'high near-natural areas',                               
    'high open public spaces',                                      
    'better air quality','high avg temperature', 'high sunshine'                                            
      
                ]},

    'Accessibility':{
        'icon': 'delivery.png',
        'buttons': [
    'highways',                                 
    'airports',                                  
    'railway stations',  
    'public transport',  
    'medical care',                         
    'pharmacy',                         
    'elementary school',                     
    'supermarkt', 'Broadband access','less road accidents']} 
        }

    # Define the number of columns for buttons
    num_columns = 10
    # Add a flag variable to keep track of recommendations generation
    recommendations_generated = False

    # Display buttons within each category
    for category, data in categories.items():
        col1, col2 = st.columns([1, 9])
        col1.image(data['icon'], width=30)
        col2.markdown(f"<h5 style='display: flex; align-items: center; margin-bottom: -30px; margin-left: -50px;'>{category}</h5>", unsafe_allow_html=True)
        col_index = 0
        col = st.columns(num_columns)
        for button in data['buttons']:
            if col_index % num_columns == 0:
                col_index = 0
            if col[col_index].button(button, key=button):
                if button not in selected_features:
                    selected_features.append(button)
                    selected_features_placeholder.text(f"Selected Features: {', '.join(selected_features)}")
                if button not in st.session_state['selected_features']:
                    st.session_state['selected_features'].append(button)
                    st.write(f"{button} selected!")
                else:
                    st.warning(f"{button} is already selected.")
            col_index += 1
        selected_features_text = ", ".join(selected_features)
        selected_features_placeholder.text(f"Selected Features: {selected_features_text}")

    # Display selected features
    st.sidebar.write("Selected Features:")
    for feature in st.session_state['selected_features']:
        st.sidebar.write(feature)

    # Check if at least two features are selected
    if len(st.session_state['selected_features']) < 2:
        st.sidebar.warning('Please select at least 2 features.')
    else:
        reset_button = st.sidebar.button("Reset")
        if not reset_button and not recommendations_generated:
            submit_button = st.sidebar.button("Submit")
                                # Allow adding new features


        # Generate recommendations if submit button is clicked
        if submit_button:
            if len(st.session_state['selected_features']) < 2:
                st.sidebar.warning('Please select at least 2 features.')
            else:
                # Create multiple pages using Streamlit's sidebar
                pages = {
                    'Generate Recommendations': generate_recommendations,
                    'About': about_page
                }

                # Add a sidebar to switch between pages
                st.sidebar.title('Navigation')
                page_selection = st.sidebar.radio('Go to', list(pages.keys()), index=0)

                if page_selection == 'Generate Recommendations':
                    city_recommendations = generate_recommendations(landkreise_scaled, st.session_state['selected_features'])
                    st.write(city_recommendations)
                    recommendations_generated = True
                    return city_recommendations
                else:
                    selected_page = pages[page_selection]
                    selected_page()

                st.spinner("Generating recommendations...")
                progress_text = "Operation in progress. Please wait."
                with st.empty():
                    my_bar = st.sidebar.progress(0, text=progress_text)
                    for percent_complete in range(100):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1, text=progress_text)

    # Reset the selected features and recommendations if reset button is clicked
    if recommendations_generated:
        reset_button = st.sidebar.button("Reset")
        if reset_button:
                        # Display selected features
            st.sidebar.write("Selected Features:")
            for feature in st.session_state['selected_features']:
                del st.session_state['selected_features']
            recommendations_generated = False
            # Clear the selected features and reset the recommendations_generated flag
            selected_features = []
            #selected_features_placeholder.text("Selected Features:")
            selected_features_placeholder = st.empty() # Reset the display of selected features
    
            st.sidebar.empty()  # Clear the sidebar content
            st.experimental_rerun()

if __name__ == '__main__':
    
    main()