import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, Polygon
from shapely.ops import split
import numpy as np
import os
import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import TopologicalError
from shapely.ops import linemerge, unary_union
from tqdm import tqdm

def buffer_centerline(input_gpkg, buffer_distance, output_gpkg = None):
    """
    Buffers the centerline network by a specified amount and saves the result to a new GeoPackage file.
    
    Parameters:
    input_gpkg (str): Path to the input GeoPackage file containing the centerline network.
    layer_name (str): Name of the layer in the GeoPackage containing the centerline network.
    output_gpkg (str): Path to the output GeoPackage file where the buffered result will be saved.
    buffer_distance (float): Buffer distance in the units of the centerline's coordinate system.
    """
    # Load the centerline network from the GeoPackage
    centerline = gpd.read_file(input_gpkg)

    # Buffer the centerline by the specified distance
    buffered_centerline = centerline.copy()
    buffered_centerline['geometry'] = centerline.geometry.buffer(buffer_distance)

    # Save the buffered centerline to a new GeoPackage file
    if output_gpkg is not None:
            # Ensure output directory exists
        os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)
        buffered_centerline.to_file(output_gpkg, driver='GPKG')
        print(f"Buffered centerline saved to {output_gpkg}")
    
    return buffered_centerline

def create_smooth_perpendicular_lines(centerline_path, line_length=60, spacing=5, window=200, output_path=None):
    # Load the centerline from the geopackage
    gdf = gpd.read_file(centerline_path)
    
    # Initialize an empty list to store perpendicular lines
    perpendiculars = []
    
    # Iterate through each feature in the GeoDataFrame
    for _, row in gdf.iterrows():
        geometry = row['geometry']
        
        # Handle MultiLineString appropriately using `geoms`
        if isinstance(geometry, MultiLineString):
            line_parts = geometry.geoms
        else:
            line_parts = [geometry]

        # Process each line part
        for line in line_parts:
            length = line.length
            num_samples = int(np.floor(length / spacing))
            for i in range(num_samples + 1):
                # Calculate the point at each meter
                point = line.interpolate(i * spacing)
                
                # Get points 20 meters ahead and behind
                point_back = line.interpolate(max(0, i * spacing - window))
                point_forward = line.interpolate(min(length, i * spacing + window))
                
                # Calculate vectors to these points
                dx_back, dy_back = point.x - point_back.x, point.y - point_back.y
                dx_forward, dy_forward = point_forward.x - point.x, point_forward.y - point.y
                
                # Average the vectors
                dx_avg = (dx_back + dx_forward) / 2
                dy_avg = (dy_back + dy_forward) / 2
                
                # Calculate the perpendicular vector
                len_vector = np.sqrt(dx_avg**2 + dy_avg**2)
                perp_vector = (-dy_avg, dx_avg)
                
                # Normalize and scale the vector
                perp_vector = (perp_vector[0] / len_vector * line_length, perp_vector[1] / len_vector * line_length)
                
                # Create the perpendicular line segment
                perp_line = LineString([
                    (point.x + perp_vector[0], point.y + perp_vector[1]),
                    (point.x - perp_vector[0], point.y - perp_vector[1])
                ])
                
                # Append the perpendicular line to the list
                perpendiculars.append({'geometry': perp_line})
    
    # Convert list to GeoDataFrame
    perpendiculars_gdf = gpd.GeoDataFrame(perpendiculars, crs=gdf.crs)
    
    # Save the perpendicular lines to the same geopackage
    if output_path is not None:
        perpendiculars_gdf.to_file(output_path, driver='GPKG')
    return perpendiculars_gdf

def create_points_along_line(centerline_path, spacing=5, output_path=None):
    # Load the centerline from the GeoPackage
    gdf = gpd.read_file(centerline_path)
    
    # Initialize an empty list to store points
    points = []
    
    # Iterate through each feature in the GeoDataFrame
    for _, row in gdf.iterrows():
        geometry = row['geometry']
        
        # Handle MultiLineString appropriately using `geoms`
        if isinstance(geometry, MultiLineString):
            line_parts = geometry.geoms
        else:
            line_parts = [geometry]

        # Process each line part
        for line in line_parts:
            length = line.length
            num_samples = int(np.floor(length / spacing))
            
            # Generate points along the line at regular intervals
            for i in range(num_samples + 1):
                # Calculate the point at each interval
                point = line.interpolate(i * spacing)
                
                # Append the point as a geometry
                points.append({'geometry': point})
    
    # Convert list to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(points, crs=gdf.crs)
    
    # Save the points to the same GeoPackage if output path is provided
    if output_path is not None:
        points_gdf.to_file(output_path, driver='GPKG')
    
    return points_gdf

import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import Polygon, MultiLineString, LineString, GeometryCollection
from shapely.errors import TopologicalError

def segment_polygon_by_lines(polygon_gpkg: str, lines_gpkg: str, output_gpkg: str):
    """
    Segments a polygon by intersecting lines.
    
    Parameters:
    polygon_gpkg (str): Path to the input GeoPackage containing the polygon.
    lines_gpkg (str): Path to the input GeoPackage containing the lines.
    output_gpkg (str): Path to the output GeoPackage for storing the segmented polygons.
    """
    
    try:
        # Read polygon and lines layers from the respective GeoPackages
        if isinstance(polygon_gpkg, gpd.GeoDataFrame):
            polygon_gdf = polygon_gpkg
        else:
            polygon_gdf = gpd.read_file(polygon_gpkg)
        
        if isinstance(lines_gpkg, gpd.GeoDataFrame):
            lines_gdf = lines_gpkg
        else:
            lines_gdf = gpd.read_file(lines_gpkg)

        # Check if the polygon and lines layers are valid
        if polygon_gdf.empty or lines_gdf.empty:
            raise ValueError("The polygon or lines layer is empty.")
        
        # Dissolve all lines into a single geometry (it could be MultiLineString or GeometryCollection)
        lines_union = lines_gdf.unary_union

        # Check if lines_union is valid
        if lines_union.is_empty:
            raise ValueError("The lines layer has no valid geometries.")
        
        # Function to split polygons by lines or collections of lines
        def split_polygon_by_geometry(polygon, splitter):
            if isinstance(splitter, LineString):
                return ops.split(polygon, splitter)
            elif isinstance(splitter, MultiLineString):
                # Process each LineString in the MultiLineString separately
                result = [polygon]
                for line in splitter.geoms:
                    temp_result = []
                    for part in result:
                        try:
                            split_result = ops.split(part, line)
                            temp_result.extend(split_result.geoms)
                        except TopologicalError as e:
                            print(f"Error splitting polygon: {e}")
                            temp_result.append(part)
                    result = temp_result
                return result
            elif isinstance(splitter, GeometryCollection):
                # Iterate over each geometry in the GeometryCollection
                result = [polygon]
                for geom in splitter.geoms:
                    if isinstance(geom, (LineString, MultiLineString)):
                        temp_result = []
                        for part in result:
                            temp_result.extend(split_polygon_by_geometry(part, geom))
                        result = temp_result
                return result
            else:
                raise ValueError(f"Unsupported geometry type for splitting: {type(splitter)}")

        # Iterate through each polygon and segment it by the lines
        segmented_polygons = []
        for polygon in polygon_gdf.geometry:
            if isinstance(polygon, Polygon):
                try:
                    # Split the polygon by lines or collections of lines
                    split_polygons = split_polygon_by_geometry(polygon, lines_union)
                    
                    # Ensure the result is valid
                    for geom in split_polygons:
                        if isinstance(geom, Polygon):
                            segmented_polygons.append(geom)
                except TopologicalError as e:
                    print(f"TopologicalError encountered while splitting: {e}")
            else:
                print("Skipping non-polygon geometry.")
        
        # Check if we have any valid polygons to write
        if not segmented_polygons:
            raise ValueError("No valid polygons were created from the segmentation.")

        # Create a new GeoDataFrame for the segmented polygons
        segmented_gdf = gpd.GeoDataFrame(geometry=segmented_polygons, crs=polygon_gdf.crs)

        # Write the segmented polygons to the output GeoPackage
        segmented_gdf.to_file(output_gpkg, driver='GPKG')

        print(f"Segmented polygons saved to {output_gpkg}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def create_hydraulic_valleys_input_polygon(centerline_path, out_segmented_poly_path, out_perps_path=None, buffer_distance=70, segment_spacing=5):
    """
    Create a polygon that represents the valley area around a centerline.
    
    Parameters:
    -----------
    centerline_path : str
        Path to the GeoPackage or shapefile containing the centerline geometry.
    
    output_path : str
        Path to save the valley polygon as a new GeoPackage or shapefile.
    
    buffer_distance : int, optional
        Width of the valley area around the centerline.
    
    Returns:
    --------
    None
        The function saves the valley polygon to the specified output file.
    """
    if out_perps_path is None:
        out_perps_path = centerline_path.replace('.gpkg', '_perps.gpkg')
    if not os.path.exists(os.path.dirname(out_perps_path)):
        os.makedirs(os.path.dirname(out_perps_path))
    if not os.path.exists(os.path.dirname(out_segmented_poly_path)):
        os.makedirs(os.path.dirname(out_segmented_poly_path))
    CL_buffered = buffer_centerline(centerline_path, buffer_distance)
    create_smooth_perpendicular_lines(centerline_path, line_length=2*buffer_distance, spacing=segment_spacing, output_path=out_perps_path)
    #merge_multipart_lines(out_perps_path, out_perps_path)
    segment_polygon_by_lines(CL_buffered, out_perps_path, out_segmented_poly_path)

def main():
    prefix = 'MM'
    segment_spacing = 5
    max_valley_width = 70
    input_path = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\Centerlines_LSDTopo\Centerlines", f"{prefix}_clipped.gpkg")
    out_poly_path = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Segmented Valleys", 
                                 f"{prefix}_segmented_valley_{segment_spacing}m.gpkg")
    out_perps_path = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Perpendiculars", 
                                  f"{prefix}_perps_{segment_spacing}m.gpkg")

    create_hydraulic_valleys_input_polygon(input_path, out_poly_path, out_perps_path, max_valley_width, segment_spacing)
    
if __name__ == '__main__':
    main()