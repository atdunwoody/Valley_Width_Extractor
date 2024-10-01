import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import os

def load_geodata(centerline_path, perpendiculars_path, hillslope_path):
    """
    Load GeoPackages for centerline, perpendiculars, and hillslopes.

    Args:
        centerline_path (str): Path to the centerline GeoPackage.
        perpendiculars_path (str): Path to the perpendiculars GeoPackage.
        hillslope_path (str): Path to the hillslopes GeoPackage.

    Returns:
        tuple: GeoDataFrames for centerline, perpendiculars, and hillslopes.
    """
    print("Loading GeoPackages...")
    centerline = gpd.read_file(centerline_path)
    perpendiculars = gpd.read_file(perpendiculars_path)
    hillslopes = gpd.read_file(hillslope_path)
    return centerline, perpendiculars, hillslopes

def ensure_crs_consistency(centerline, perpendiculars, hillslopes):
    """
    Ensure all GeoDataFrames have the same Coordinate Reference System (CRS).
    Reprojects perpendiculars and hillslopes to match centerline CRS if necessary.

    Args:
        centerline (GeoDataFrame): Centerline GeoDataFrame.
        perpendiculars (GeoDataFrame): Perpendiculars GeoDataFrame.
        hillslopes (GeoDataFrame): Hillslopes GeoDataFrame.

    Returns:
        tuple: CRS-consistent GeoDataFrames for perpendiculars and hillslopes.
    """
    print("Checking CRS consistency...")
    if not (centerline.crs == perpendiculars.crs == hillslopes.crs):
        print("CRS mismatch detected. Reprojecting perpendiculars and hillslopes to match centerline CRS.")
        perpendiculars = perpendiculars.to_crs(centerline.crs)
        hillslopes = hillslopes.to_crs(centerline.crs)
    else:
        print("All GeoDataFrames have consistent CRS.")
    return perpendiculars, hillslopes

def get_intersection(line, center_geom):
    """
    Find the intersection point between a line and the centerline geometry.

    Args:
        line (shapely.geometry.LineString): Perpendicular line geometry.
        center_geom (shapely.geometry.Geometry): Centerline geometry.

    Returns:
        shapely.geometry.Point or None: Intersection point if exists, else None.
    """
    intersection = line.intersection(center_geom)
    if intersection.is_empty:
        return None
    elif isinstance(intersection, Point):
        return intersection
    elif intersection.geom_type == 'MultiPoint':
        # If multiple intersection points, take the first one
        return list(intersection)[0]
    else:
        return None

def find_intersection_points(perpendiculars, centerline_geom):
    """
    Find and assign intersection points between perpendiculars and the centerline.

    Args:
        perpendiculars (GeoDataFrame): Perpendiculars GeoDataFrame.
        centerline_geom (shapely.geometry.Geometry): Centerline geometry.

    Returns:
        GeoDataFrame: Filtered perpendiculars with intersection points.
    """
    print("Finding intersection points between centerline and perpendicular lines...")
    perpendiculars['intersection_point'] = perpendiculars.geometry.apply(lambda x: get_intersection(x, centerline_geom))
    
    # Remove perpendiculars that do not intersect the centerline
    initial_count = len(perpendiculars)
    perpendiculars = perpendiculars[perpendiculars['intersection_point'].notnull()].copy()
    filtered_count = len(perpendiculars)
    print(f"Filtered out {initial_count - filtered_count} perpendiculars without intersection.")
    return perpendiculars

def buffer_intersection_points(perpendiculars, buffer_distance=2):
    """
    Buffer each intersection point by a specified distance.

    Args:
        perpendiculars (GeoDataFrame): Perpendiculars GeoDataFrame with intersection points.
        buffer_distance (float, optional): Buffer distance in meters. Defaults to 2.

    Returns:
        GeoDataFrame: Perpendiculars GeoDataFrame with buffered geometries.
    """
    print(f"Buffering intersection points by {buffer_distance} meters...")
    perpendiculars['buffer_2m'] = perpendiculars['intersection_point'].buffer(buffer_distance)
    return perpendiculars

def create_hillslope_spatial_index(hillslopes):
    """
    Create a spatial index for hillslopes to optimize spatial queries.

    Args:
        hillslopes (GeoDataFrame): Hillslopes GeoDataFrame.

    Returns:
        spatial index object: Spatial index for hillslopes.
    """
    print("Creating spatial index for hillslopes...")
    return hillslopes.sindex

def find_intersecting_hillslopes(buffer_geom, hillslopes, hillslope_sindex):
    """
    Find hillslopes that intersect with a given buffer geometry.

    Args:
        buffer_geom (shapely.geometry.Geometry): Buffered geometry.
        hillslopes (GeoDataFrame): Hillslopes GeoDataFrame.
        hillslope_sindex (spatial index): Spatial index for hillslopes.

    Returns:
        GeoDataFrame: Hillslopes that intersect with the buffer geometry.
    """
    possible_matches_index = list(hillslope_sindex.intersection(buffer_geom.bounds))
    possible_matches = hillslopes.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer_geom)]
    return precise_matches

def identify_hillslopes_intersections(perpendiculars, hillslopes, hillslope_sindex):
    """
    Identify hillslopes intersecting each buffered intersection point.

    Args:
        perpendiculars (GeoDataFrame): Perpendiculars GeoDataFrame with buffered geometries.
        hillslopes (GeoDataFrame): Hillslopes GeoDataFrame.
        hillslope_sindex (spatial index): Spatial index for hillslopes.

    Returns:
        GeoDataFrame: Perpendiculars GeoDataFrame with intersecting hillslopes.
    """
    print("Identifying hillslopes intersecting each buffered point...")
    perpendiculars['intersecting_hillslopes'] = perpendiculars['buffer_2m'].apply(
        lambda x: find_intersecting_hillslopes(x, hillslopes, hillslope_sindex)
    )
    return perpendiculars

def clip_perpendicular_lines(perpendiculars, centerline_crs):
    """
    Clip each perpendicular line using the intersecting hillslopes.

    Args:
        perpendiculars (GeoDataFrame): Perpendiculars GeoDataFrame with intersecting hillslopes.
        centerline_crs (dict or CRS): Coordinate Reference System of the centerline.

    Returns:
        GeoDataFrame: Clipped perpendiculars GeoDataFrame.
    """
    print("Clipping perpendicular lines using intersecting hillslopes...")
    clipped_geometries = []
    clipped_attributes = []
    
    for idx, row in perpendiculars.iterrows():
        perp_id = row.get('id', idx)  # Replace 'id' with the actual ID field if different
        line = row.geometry
        intersecting_hillslopes = row['intersecting_hillslopes']
        
        if intersecting_hillslopes.empty:
            # If no hillslopes intersect, retain the original line
            clipped_geometries.append(line)
        else:
            # Collect all clipped parts from intersecting hillslopes
            clipped_parts = []
            for _, hill_row in intersecting_hillslopes.iterrows():
                clipped = line.intersection(hill_row.geometry)
                if clipped.is_empty:
                    continue
                elif isinstance(clipped, (LineString, MultiLineString)):
                    clipped_parts.append(clipped)
                # Handle other geometry types if necessary
            
            if clipped_parts:
                # Merge all clipped parts into a single geometry
                merged = unary_union(clipped_parts)
                clipped_geometries.append(merged)
            else:
                # If no valid clipping occurred, retain the original line
                clipped_geometries.append(line)
        
        # Append attributes (only 'perp_id' to maintain one-to-one correspondence)
        clipped_attributes.append({'perp_id': perp_id})
    
    print("Creating GeoDataFrame for clipped lines...")
    clipped_gdf = gpd.GeoDataFrame(clipped_attributes, geometry=clipped_geometries, crs=centerline_crs)
    
    # Optional: Remove duplicate geometries if any
    clipped_gdf = clipped_gdf.drop_duplicates()
    
    return clipped_gdf

def save_clipped_perpendiculars(clipped_gdf, output_gpkg):
    """
    Save the clipped perpendiculars GeoDataFrame to a GeoPackage.

    Args:
        clipped_gdf (GeoDataFrame): Clipped perpendiculars GeoDataFrame.
        output_gpkg (str): Path to the output GeoPackage.
    """
    print(f"Saving clipped perpendiculars to {output_gpkg}...")
    # If the output GeoPackage already exists, overwrite it
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)
    
    clipped_gdf.to_file(output_gpkg, driver='GPKG')
    print("Processing complete. Clipped perpendiculars have been saved.")

def clip_perpendiculars_by_hillslope(perpendiculars_path, hillslope_path, centerline_path, output_gpkg):
    
    # Step 1: Load the GeoPackages
    centerline, perpendiculars, hillslopes = load_geodata(centerline_path, perpendiculars_path, hillslope_path)
    
    # Step 2: Ensure CRS consistency
    perpendiculars, hillslopes = ensure_crs_consistency(centerline, perpendiculars, hillslopes)
    
    # Step 3: Find Intersection Points
    print("Finding intersection points between centerline and perpendicular lines...")
    centerline_geom = centerline.unary_union
    perpendiculars = find_intersection_points(perpendiculars, centerline_geom)
    
    # Step 4: Buffer Intersection Points
    perpendiculars = buffer_intersection_points(perpendiculars, buffer_distance=2)
    
    # Step 5: Identify Hillslopes Intersecting Buffered Points
    hillslope_sindex = create_hillslope_spatial_index(hillslopes)
    perpendiculars = identify_hillslopes_intersections(perpendiculars, hillslopes, hillslope_sindex)
    
    # Step 6: Clip Perpendicular Lines Using Hillslopes
    clipped_gdf = clip_perpendicular_lines(perpendiculars, centerline.crs)
    
    # Step 7: Save the Clipped Perpendiculars to the Output GeoPackage
    save_clipped_perpendiculars(clipped_gdf, output_gpkg)

if __name__ == "__main__":
    # Define input paths
    perpendiculars_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\ME\Buffered Perpendiculars\ME_multi_perps_5m.gpkg"
    hillslope_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\hillslopes_200k.gpkg"
    centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\ME_clipped.gpkg"
    
    # Define output path
    output_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\ME\Buffered Perpendiculars\ME_multi_perps_hillslope_5m.gpkg"
    clip_perpendiculars_by_hillslope(perpendiculars_path, hillslope_path, centerline_path, output_gpkg)
