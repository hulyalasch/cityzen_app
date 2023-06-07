#STREAMLIT APP


import streamlit as st
import pandas as pd
import numpy as np
# Set the page icon and layout type
st.set_page_config(page_title="CityZen App", page_icon="location.png", layout="centered")

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
#from pages.generate_recommendations import generate_recommendations
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

geometry = gpd.read_file("geometry_23.topojson")

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
 

#@st.cache_data(experimental_allow_widgets=True)
# Create a Streamlit app
def main():
    
    # Set the app title
    st.title('CityZen')
    st.sidebar.header("Find your Zen City")

    st.markdown(f"<h3 style='display: flex; align-items: center; margin-bottom: 0px; margin-left: 0px;'>To find your ideal Zen City, what are the essential characteristics you value the most?</h3>", unsafe_allow_html=True)

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
    categories = {    'Accessibility':{
        'icon': 'delivery.png',
        'buttons': [
    'highways',                                 
    'airports',                                  
    'railway stations',  
    'public transport',  
    'medical care',                         
    'pharmacy',                         
    'elementary school',                     
    'supermarkt', 'Broadband access','less road accidents']},
        'Affordability': {
        'icon': 'funding.png',
        'buttons': [ 'low unemployment',
    'low land prices',                                                
    'low rental prices',                                                    
    'high income',                                              
    'high purchasing power',  'more business places']},
            'Social': {
        'icon': 'social.png',
        'buttons': [ 'Child care friendly', 'low avg age', 'high avg age',
    'high cultural diversity', 'high tourism', 'low tourism', 'more restaurants', 'more shopping malls', 'high capacity of hospitals',                                            
    'more associations',  'more sports and leisure activities']},
            'Environmental': {
        'icon': 'city.png',
        'buttons':[ 'low industry', 'high industry',                                            
    'low rurality', 'high rurality', 'low population density',                                              
       'high coastal area',                                                 
    'high green spaces',                                 
    'high near-natural areas',                               
    'high open public spaces',                                      
    'better air quality','high avg temperature', 'high sunshine']}
        }

    # Define the number of columns for buttons
    num_columns = 6
    # Add a flag variable to keep track of recommendations generation
    recommendations_generated = False
    

    # Display buttons within each category
    for category, data in categories.items():
        col1, col2 = st.columns([1, 9])
        col1.image(data['icon'], width=30)
        col2.markdown(f"<h5 style='display: flex; align-items: center; margin-bottom: -30px; margin-left: -40px;'>{category}</h5>", unsafe_allow_html=True)
        col_index = 0
        col = st.columns(num_columns)
        for button in data['buttons']:
            if col_index % num_columns == 0:
                col_index = 0
            if col[col_index].button(button, key=button):
                if button not in selected_features:
                    selected_features.append(button)
                    selected_features_placeholder.text(f"You can select multiple parameters: {', '.join(selected_features)}")
                if button not in st.session_state['selected_features']:
                    st.session_state['selected_features'].append(button)
                    st.write(f"{button} selected!")
                else:
                    st.warning(f"{button} is already selected.")
            col_index += 1
        selected_features_text = ", ".join(selected_features)
        selected_features_placeholder.text(f"You can select multiple parameters: {selected_features_text}")

    # Initialize variables
    submit_button = False
    recommendations_generated = False
    # Display selected features
    st.sidebar.write("Selected Features:")
    submit_button = False
    recommendations_generated = False
    for feature in st.session_state['selected_features']:
        st.sidebar.write(feature)

    # Check if at least two features are selected
    if len(st.session_state['selected_features']) < 2:
        st.sidebar.warning('Please select at least 2 features.')
    else:
        reset_button = st.sidebar.button("Reset")
        if not reset_button and not recommendations_generated:
            submit_button = st.sidebar.button("Submit")
            st.sidebar.warning("To initiate a complete reset, please refresh the webpage.")
                                # Allow adding new features


    # Generate recommendations if submit button is clicked
    if submit_button:
        if len(st.session_state['selected_features']) < 2:
            st.sidebar.warning('Please select at least 2 features.')
        else:
            st.spinner("Generating recommendations...")
            progress_text = "Operation in progress. Please wait."
            with st.empty():
                my_bar = st.sidebar.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                    st.success("Operation completed successfully!")
            city_recommendations = generate_recommendations(landkreise_scaled, st.session_state['selected_features'])
            #st.write(city_recommendations)
            recommendations_generated = True
                    # Display balloons animation
                    # Display a success message
            st.success("Operation completed successfully!")
            st.balloons()
            return city_recommendations


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
        

#@st.cache_data(experimental_allow_widgets=True)               
# Generate recommendations based on selected features
def generate_recommendations(landkreise_scaled, selected_features):
    # Select the columns of interest from the data
    selected_data = landkreise_scaled[selected_features]

    # Calculate the cosine similarity matrix based on the selected columns
    similarity_matrix = cosine_similarity(selected_data)

    city_names = landkreise_scaled.index
    #num_cities = len(city_names)
    num_cities = similarity_matrix.shape[0]

    # Generate recommendations for each city
    city_recommendations = {}
    recommendations_gdfs ={}
    texts=[]
    # Create a list to store the Wikipedia links for each recommended city
    wiki_links = []
    recommended_city_urls = []


    for i in range(num_cities):
        # Retrieve the similarity scores for the current city
        similarity_scores = similarity_matrix[i]

        # Sort the similarity scores in descending order and get the indices of the top 10 cities (excluding the current city itself)
        top_indices = similarity_scores.argsort()[::-1][1:11]

        # Map the indices back to city names to get the top 10 recommendations
        top_cities = [city_names[j] for j in top_indices]

        # Create a new GeoDataFrame for the current city and its recommendations
        city_recommendations = geometry.loc[top_cities].copy()

        # Store the recommendations GeoDataFrame for the current city
        recommendations_gdfs[city_names[i]] = city_recommendations

        # Add the recommended city names to the recommendations_gdfs dictionary
        recommended_city_names = city_recommendations['gen'].tolist()
        recommendations_gdfs[city_names[i]]['recommended_cities'] = recommended_city_names

        # Iterate over the recommended city names and retrieve the Wikipedia links
    
    # Create a sidebar
    st.sidebar.title('Recommended Cities')
    wiki_links = []

    for city_name1 in recommended_city_names:
        wiki_url = "https://de.wikipedia.org/wiki/" + city_name1.replace(" ", "-")
        recommended_city_urls.append(wiki_url)

        # Extract just the city name from the Wikipedia page title
        city_name = city_name1.split(':')[0]

        # Display the city name as a blue link
        st.sidebar.markdown(f"[{city_name}]({wiki_url})")

        # Update the 'Wikipedia Link' column in recommendations_gdfs for the current city
        recommendations_gdfs[city_names[i]].loc[recommendations_gdfs[city_names[i]]['gen'] == city_name1, 'Wikipedia Link'] = wiki_url

        # Append the wiki_url to the wiki_links list
        wiki_links.append(wiki_url)

    # Display the recommended cities as a comma-separated list
    #st.write(f"Recommended cities: {', '.join(wiki_links)}")

        # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the map of Germany on the axis
    geometry.plot(ax=ax, color='#a4a4a6')

        # Access the geometry column of the selected cities
    city_polygons = city_recommendations['geometry']

        # Plot the city polygons on the axis
    city_polygons.plot(ax=ax, color='#f2eb80', edgecolor='#80807e')

        # Customize the plot
    ax.set_title('Recommended Cities in Germany')
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    ax.axis('off')


    
    # Iterate over the city polygons and add labels
    for idx, city_polygon in enumerate(city_polygons):
        city_name = recommended_city_names[idx]  # Adjust index to match recommended_city_names
        centroid = city_polygon.centroid
        text = ax.annotate(city_name, (centroid.x, centroid.y), ha='center', va='center', fontsize=8)
        texts.append(text)
        # Adjust the position of labels to avoid overlapping
    adjustText.adjust_text(texts, ax=ax)

        # Display the plot
    # Display the map on Streamlit
    # Display the plot
    st.pyplot(fig)

    st.title('Maps of Selected parameters')

    # Plot a map for each selected feature
    selected_features = st.session_state['selected_features']
    num_columns = len(selected_features)

    # Create columns for each map
    columns = st.columns(num_columns)

    for i, feature in enumerate(selected_features):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the column with a legend
        geometry.plot(ax=ax, column=feature, legend=True)

        # Add a title to the plot
        plt.title(f"Map of {feature}", fontsize=30)

        ax.axis('off')

        # Display the plot in the respective column
        with columns[i]:
            st.pyplot(fig)

if __name__ == '__main__':
    
    main()