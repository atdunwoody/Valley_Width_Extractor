import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import os

# Define input paths
perpendiculars_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\UE_clipped_perps_5m.gpkg"
hillslope_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\hillslopes.gpkg"
centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\UE_clipped.gpkg"

# Define output path
output_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\UE_clipped_perps_5m_hillslopes.gpkg"

# Step 1: Load the GeoPackages
print("Loading GeoPackages...")
centerline = gpd.read_file(centerline_path)
perpendiculars = gpd.read_file(perpendiculars_path)
hillslopes = gpd.read_file(hillslope_path)

# Ensure all layers have the same Coordinate Reference System (CRS)
print("Checking CRS consistency...")
if not (centerline.crs == perpendiculars.crs == hillslopes.crs):
    print("CRS mismatch detected. Reprojecting perpendiculars and hillslopes to match centerline CRS.")
    perpendiculars = perpendiculars.to_crs(centerline.crs)
    hillslopes = hillslopes.to_crs(centerline.crs)

# Step 2: Find Intersection Points between Centerline and Perpendiculars
print("Finding intersection points between centerline and perpendicular lines...")
# Assuming the centerline layer has a single feature
centerline_geom = centerline.unary_union

# Function to find the intersection point between a line and the centerline
def get_intersection(line, center_geom):
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

# Apply the intersection function to each perpendicular line
perpendiculars['intersection_point'] = perpendiculars.geometry.apply(lambda x: get_intersection(x, centerline_geom))

# Remove perpendiculars that do not intersect the centerline
initial_count = len(perpendiculars)
perpendiculars = perpendiculars[perpendiculars['intersection_point'].notnull()].copy()
filtered_count = len(perpendiculars)
print(f"Filtered out {initial_count - filtered_count} perpendiculars without intersection.")

# Step 3: Buffer Each Intersection Point by 2 Meters
print("Buffering intersection points by 2 meters...")
perpendiculars['buffer_2m'] = perpendiculars['intersection_point'].buffer(2)

# Step 4: Identify Hillslopes Intersecting Each Buffered Point
print("Identifying hillslopes intersecting each buffered point...")
# Create a spatial index for hillslopes for faster querying
hillslope_sindex = hillslopes.sindex

def find_intersecting_hillslopes(buffer_geom):
    possible_matches_index = list(hillslope_sindex.intersection(buffer_geom.bounds))
    possible_matches = hillslopes.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer_geom)]
    return precise_matches

# Apply the function to find intersecting hillslopes for each buffer
perpendiculars['intersecting_hillslopes'] = perpendiculars['buffer_2m'].apply(find_intersecting_hillslopes)

# Step 5: Clip Each Perpendicular Line Using the Intersecting Hillslopes
print("Clipping perpendicular lines using intersecting hillslopes...")
# Prepare lists to collect clipped geometries and their attributes
clipped_geometries = []
clipped_attributes = []

# Iterate over each perpendicular line
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

# Create a GeoDataFrame from the clipped geometries
print("Creating GeoDataFrame for clipped lines...")
clipped_gdf = gpd.GeoDataFrame(clipped_attributes, geometry=clipped_geometries, crs=centerline.crs)

# Optional: Remove duplicate geometries if any
clipped_gdf = clipped_gdf.drop_duplicates()

# Step 6: Save the Clipped Perpendiculars to the Output GeoPackage
print(f"Saving clipped perpendiculars to {output_gpkg}...")
# If the output GeoPackage already exists, overwrite it
if os.path.exists(output_gpkg):
    os.remove(output_gpkg)

clipped_gdf.to_file(output_gpkg, driver='GPKG')

print("Processing complete. Clipped perpendiculars have been saved.")
