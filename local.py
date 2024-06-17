import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
import streamlit as st
from streamlit_folium import folium_static as st_folium
import numpy as np
from scipy.spatial import ConvexHull
import google.generativeai as genai
from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import io
from PIL import Image

headers ={
    "authorization": st.secrets["API_KEY"],
    "content-type": "application/json"
}

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Read the data into a DataFrame
@st.cache_resource
def load_data():
    return pd.read_csv('Cleaned Lat Long/datasetk.csv')

def fetch_and_update_data(year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap):
    df = load_data()
    update_heatmap(df, year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap)

def give_analysis():
    genai.configure(api_key=headers["authorization"])
    # Set up the model
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                  generation_config=generation_config,
                                  )

    image_path = Path("image.jpeg")

    prompt= "Based on the image given about Blackspot analytics in the Karnataka India region. Assume the role of a data analyst. You have to present a police officer with the key observations and useful insights and analytics that will aid in decision-making to improve resource utilization, traffic management, junction control, traffic signal optimization, police bandobast, etc. Also, list all the hotspots (orange marker clusters) and predicted hotspots represented in black markers. "
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes()
    }

    prompt_parts = [
        prompt, image_part
    ]
    response = model.generate_content(prompt_parts)
    print(response.text)
    return response.text

def save_map_as_image(m):
    img_data = m._to_png(1)  # Get the map image data as PNG
    img = Image.open(io.BytesIO(img_data))  # Open the image data
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # Convert to RGB mode if necessary
    img.save('image.jpeg')  # Save as JPEG

def update_heatmap(df, year_range, Collision_Type, Severity, month_range, District, Accident_Classification, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap):
    
    filtered_df = df[(df['Year_x'].between(year_range[0], year_range[1])) & 
                     (df['Collision_Type'].isin(Collision_Type)) & 
                     (df['Severity'].isin(Severity)) & 
                     (df['Month'].between(month_range[0], month_range[1])) & 
                     (df['DISTRICTNAME'] == District) &
                     (df['Accident_Classification'].isin(Accident_Classification))]

    if filtered_df.empty:
        # Create an empty map
        st.warning("No data available for the selected filters.")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=6)
    else:
        locations = filtered_df[['Latitude_x', 'Longitude_x']].values
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.01, min_samples=min_samples)
        filtered_df['cluster'] = dbscan.fit_predict(locations)
        
        # Group by cluster and count the number of points in each cluster
        cluster_counts = filtered_df.groupby('cluster').size().reset_index(name='count')
        
        # Filter clusters based on the number of points
        dense_clusters = cluster_counts[cluster_counts['count'] >= num_markers]
        
        # Sort clusters by count in descending order and select top 5 dense clusters
        top_dense_clusters = dense_clusters.nlargest(num_markers, 'count')
        
        m = folium.Map(location=[filtered_df['Latitude_x'].mean(), filtered_df['Longitude_x'].mean()], zoom_start=10)
        
        # Add blackspots
        if show_blackspots:
            for index, row in top_dense_clusters.iterrows():
                cluster_df = filtered_df[filtered_df['cluster'] == row['cluster']]
                points = cluster_df[['Latitude_x', 'Longitude_x']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_centroid = np.mean(hull_points, axis=0)
                hull_lat, hull_lon = hull_centroid[0], hull_centroid[1]
                
                # Find the nearest data point to the hull centroid
                nearest_point_idx = np.argmin(np.sum((points - hull_centroid) ** 2, axis=1))
                nearest_point = points[nearest_point_idx]
                nearest_lat, nearest_lon = nearest_point[0], nearest_point[1]
                
                # Calculate average Collision Type and Severity for the nearest point
                avg_collision_type = cluster_df.iloc[nearest_point_idx]['Collision_Type']
                avg_severity = cluster_df.iloc[nearest_point_idx]['Severity']
                
                # Construct popup content with average information
                popup_content = f"Density: {row['count']}<br>"
                popup_content += f"Collision Type: {avg_collision_type}<br>"
                popup_content += f"Severity: {avg_severity}<br>"
                
                folium.Marker(location=[nearest_lat, nearest_lon], popup=popup_content, icon=folium.Icon(color='black')).add_to(m)

        # Add heatmap layer
        if show_heatmap:
            heat_map = HeatMap(locations)
            m.add_child(heat_map)

        # Add marker cluster for individual accidents
        if show_marker_cluster:
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add individual markers to the MarkerCluster layer
            locations_markers = filtered_df[['Latitude_x', 'Longitude_x']].values.tolist()
            for loc in locations_markers:
                folium.Marker(loc).add_to(marker_cluster)

    # Display the map
    if m is not None:
        st_folium(m, width=1410, height=750)
            # Save the map as an image
        save_map_as_image(m)
    # Perform the analysis
    texttt = give_analysis()

    # Display spinner while analysis is being computed
    with st.spinner("Getting analysis..."):
        # Perform the analysis
        texttt = give_analysis()

        # Define CSS for the card-like structure
        card_style = """
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            background-color: #f7f7f7;
        """

        # Display the analysis result
        st.markdown(
            f'<div style="{card_style}">{texttt}</div>', 
            unsafe_allow_html=True
        )

df = load_data()
st.markdown("<h1 style='text-align: center; color: black;'>Blackspot Analysis & Prediction</h1>", unsafe_allow_html=True)

# Sidebar widgets for selecting filters and layers
st.sidebar.markdown("<h1 style='text-align: left; font-size: 35px;'>Slicers</h1>", unsafe_allow_html=True)

show_heatmap = st.sidebar.checkbox('Show Heatmap', value=False)
show_marker_cluster = st.sidebar.checkbox('Show Marker Cluster', value=True)
show_blackspots = st.sidebar.checkbox('Show Predicted Blackspots', value=True)

year_range = st.sidebar.slider(
    '**Select a range of years:**',
    int(df['Year_x'].min()), int(df['Year_x'].max()), (int(df['Year_x'].min()), int(df['Year_x'].max()))
)

month_range = st.sidebar.slider(
    '**Select a range of months:**',
    int(df['Month'].min()), int(df['Month'].max()), (int(df['Month'].min()), int(df['Month'].max()))
)
Accident_Classification_checkbox = st.sidebar.multiselect(
    'Accident Classification:',
    df['Accident_Classification'].unique()
)

Collision_Type_checkbox = st.sidebar.multiselect(
    'Collision Type:',
    df['Collision_Type'].unique()
)

Severity_checkbox = st.sidebar.multiselect(
    'Severity:',
    df['Severity'].unique()
)

District_dropdown = st.sidebar.selectbox(
    'District:',
    df['DISTRICTNAME'].unique()
)

num_markers = st.sidebar.slider(
    'Number of Blackspots:',
    1, 15, 8
)

min_samples = st.sidebar.slider(
    'Minimum Number of Neighbors for Density:',
    50, 200, 75
)

# Button to fetch data
if st.sidebar.button('Fetch Data'):
    fetch_and_update_data(year_range, Collision_Type_checkbox, Severity_checkbox, month_range, District_dropdown, Accident_Classification_checkbox, num_markers, min_samples, show_blackspots, show_marker_cluster, show_heatmap)
    give_analysis()  # You can call give_analysis() here if needed