#STREAMLIT APP


import streamlit as st
import pandas as pd
import numpy as np
# Set the page icon and layout type
st.set_page_config(page_title="CityZen App", page_icon="location.png", layout="wide")

from numpy import array
# Example: Importing specific data types from NumPy
from numpy import int32, float64
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
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

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

#st.success('This is a success message!', icon="✅")


# Initialize Wikipedia API
#wiki_api = wikipediaapi.Wikipedia('en')

geometry = gpd.read_file("geometry_7.geojson")

landkreise_scaled = pd.read_csv("landkreise_scaled.csv")

# Define the list of 20 features
all_features = landkreise_scaled.columns.to_list()[1:]

# Create a Streamlit app
def main():
    
    # Set the app title
    st.title('CityZen')
    st.sidebar.title("Find your Zen City")

    st.sidebar.header("Which characteristics of cities are most important to you when selecting your place of residence?")

    # Select 6 prioritized features
    #st.header('Select 6 Prioritized Features')
    # Create an empty placeholder to hold the selected features
    selected_features_placeholder = st.empty()
    
    selected_features = []


    # Display all features as buttons
    for feature in all_features:
        if st.sidebar.button(feature, key=feature):
            selected_features.append(feature)
            selected_features_placeholder.text(f"Selected Features: {', '.join(selected_features)}")
            if feature not in selected_features:
                selected_features.append(feature)
    
                # Update the selected features placeholder
        selected_features_placeholder.text(f"Selected Features: {', '.join(selected_features)}")

            # Check if at least three parameters are selected
        if len(selected_features) < 3:
            st.warning('Please select at least three parameters.')
    else:

        try:    
            # Generate recommendations based on selected features
            with st.spinner("Generating recommendations..."):
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(5, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)

            city_recommendations = generate_recommendations(landkreise_scaled, selected_features)
                                # Delay before showing the progress message
                
            return city_recommendations

        except ValueError:
            # Handle the case when no features are selected
            st.warning('Please select at least one parameter to generate recommendations.')

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
        st.markdown(f"[{city_name}]({wiki_url})")

        # Update the 'Wikipedia Link' column in recommendations_gdfs for the current city
        recommendations_gdfs[city_names[i]].loc[recommendations_gdfs[city_names[i]]['gen'] == city_name1, 'Wikipedia Link'] = wiki_url

        # Append the wiki_url to the wiki_links list
        wiki_links.append(wiki_url)

    # Display the recommended cities as a comma-separated list
    #st.write(f"Recommended cities: {', '.join(wiki_links)}")


    print("Recommendations for", selected_features)
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
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    
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
    for feature in selected_features:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the column with a legend
        geometry.plot(ax=ax, column=feature, legend=True)

        # Add a title to the plot
        plt.title(f"Map of {feature}")

        # Display the plot
        st.pyplot(fig)
     

    #return city_recommendations

if __name__ == '__main__':
    
    main()