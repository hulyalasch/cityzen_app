import streamlit as st


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
add_bg_from_local('cityzen_24.jpg') 


# Function for the "About" page
def about_page():

    st.title('About')
    # Add content to the sidebar
    st.sidebar.markdown("Made with ❤️ by [hulyalasch](https://github.com/hulyalasch)")
    st.sidebar.markdown("If you have any questions or would like to connect, please don't hesitate to reach out to me on [LinkedIn](https://www.linkedin.com/in/h%C3%BClya-lasch/)")

    st.write('CityZen App is a Streamlit app for generating recommendations for cities in Germany based on selected parameters. CityZen App aims to help users find their ideal city based on their preferences by considering multiple parameters.')

    st.markdown('### How the CityZen App works?')
    st.write('- Select specific parameters to calculate similarity scores')
    st.write('- Generate recommendations for selected parameters based on similarity scores of cities')
    st.write('- Visualize the recommended cities on a map of Germany')
    st.write('- Display maps of selected parameters for all cities')

    st.markdown('### Data')
    st.write('The data used in this app consists of information about different cities in Germany, including various parameters such as population, socio-economic factors, infrastructure, etc.')
    st.write('The parameters are normalized and used only population-based values or ratios to provide comparability.' )

    st.markdown('### Data Sources')
    st.write('INKAR BBSR : INKAR - Indicators and Maps of Spatial and Urban Development, © Bundesinstitut für Bau-, Stadt und Raumforschung, 2023 https://www.inkar.de/' )
    st.write('INFAS 360, Corona Datenplattform (Healthcare Datenplattform): The data collection on this platform is based on the project of the Federal Ministry of Economy and Climate Protection (BMWK) during the Corona pandemic (2020-2022). https://www.healthcare-datenplattform.de/' )


    st.markdown('### References')
    st.write('This app utilizes the following libraries and frameworks:')
    st.write('- Streamlit: https://streamlit.io/')
    st.write('- Matplotlib: https://matplotlib.org/')
    st.write('- adjustText: https://github.com/Phlya/adjustText')
    st.write('- scikit-learn cosine similarity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html')
    st.write('- scikit-learn scaling: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html')
    st.write('- Simplifying the geometries: https://mapshaper.org/, https://github.com/yasserius/mapshaper_geojson_simplify')

    

# Uncomment the following line if you want to test the "About" page independently
about_page()