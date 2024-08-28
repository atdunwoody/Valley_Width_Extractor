import geopandas as gpd
from shapely.geometry import LineString
def filter_stream_order(input_geojson_path, output_gpkg_path, stream_order_threshold=4):
    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(input_geojson_path)
    
    # Check if 'StreamOrder' column exists
    if 'StreamOrder' not in gdf.columns:
        raise ValueError("The GeoJSON file does not contain a 'StreamOrder' column.")
    
    # Filter the GeoDataFrame to keep only points with StreamOrder >= stream_order_threshold
    filtered_gdf = gdf[gdf['StreamOrder'] >= stream_order_threshold]
    
    # Save the filtered GeoDataFrame to a GeoPackage
    filtered_gdf.to_file(output_gpkg_path, driver="GPKG")


import geopandas as gpd
from shapely.geometry import LineString

def connect_points_to_lines(input_gpkg_path, output_gpkg_path):
    # Read the GeoPackage into a GeoDataFrame
    gdf = gpd.read_file(input_gpkg_path)
    
    # Create a dictionary to store points by their 'NI' value
    ni_dict = {row['NI']: (row.geometry, row['receiver_JI']) for idx, row in gdf.iterrows()}
    
    # List to hold the LineString geometries and receiver_JI values
    lines = []
    receiver_jis = []
    
    # Iterate through the GeoDataFrame
    for idx, point in gdf.iterrows():
        ni = point['NI']
        receiver_ni = point['receiver_NI']
        
        # Ensure the current point's receiver_NI matches another point's NI
        if receiver_ni in ni_dict:
            # Create a LineString from the current point to the downstream point
            line = LineString([ni_dict[ni][0], ni_dict[receiver_ni][0]])
            lines.append(line)
            receiver_jis.append(ni_dict[ni][1])  # Append the receiver_JI of the line
    
    # Create a new GeoDataFrame with the LineString geometries and receiver_JI
    lines_gdf = gpd.GeoDataFrame({'geometry': lines, 'receiver_JI': receiver_jis}, crs=gdf.crs)
    
    # Save the resulting GeoDataFrame to a GeoPackage
    lines_gdf.to_file(output_gpkg_path, driver="GPKG")

import geopandas as gpd
from shapely.ops import linemerge
from shapely.geometry import MultiLineString

def merge_lines_by_receiver_ji(input_gpkg_path, output_gpkg_path):
    # Read the GeoPackage into a GeoDataFrame
    gdf = gpd.read_file(input_gpkg_path)
    
    # Check if the 'receiver_JI' field exists
    if 'receiver_JI' not in gdf.columns:
        raise ValueError("The input GeoPackage does not contain a 'receiver_JI' field.")
    
    # Group the line segments by 'receiver_JI'
    grouped = gdf.groupby('receiver_JI')
    
    # List to hold merged LineStrings and corresponding receiver_JI values
    merged_lines = []
    receiver_jis = []
    
    # Iterate through each group and merge the line segments
    for receiver_ji, group in grouped:
        # Convert the array of geometries into a list
        lines = list(group.geometry)
        
        # Merge the line segments into a single LineString or MultiLineString
        merged_line = linemerge(lines)
        
        # Ensure the result is a LineString or MultiLineString
        if isinstance(merged_line, MultiLineString):
            merged_line = linemerge(merged_line)
        
        # Append the merged line and the receiver_JI value
        merged_lines.append(merged_line)
        receiver_jis.append(receiver_ji)
    
    # Create a new GeoDataFrame with the merged LineString geometries and receiver_JI
    merged_gdf = gpd.GeoDataFrame({'geometry': merged_lines, 'receiver_JI': receiver_jis}, crs=gdf.crs)
    
    # Save the resulting GeoDataFrame to a GeoPackage
    merged_gdf.to_file(output_gpkg_path, driver="GPKG")


input_gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\Bennett_Centerline_Points.gpkg"
output_gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\Bennett_Centerlines.gpkg"
connect_points_to_lines(input_gpkg_path, output_gpkg_path)
merge_lines_by_receiver_ji(output_gpkg_path, output_gpkg_path)


# input_geojson_path = r"C:\LSDTopoTools\Bennett\2021_LIDAR_Bennett_Clip_CN.geojson"
# output_gpkg_path = r"C:\LSDTopoTools\Bennett\2021_LIDAR_Bennett_Clip_CN_pruned.geojson"

# filter_stream_order(input_geojson_path, output_gpkg_path)
