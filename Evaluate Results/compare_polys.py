import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Input GeoPackage paths
reference_gpkg = "path_to_reference.gpkg"  # Replace with the path to your reference GeoPackage
source_gpkg = "path_to_source.gpkg"        # Replace with the path to your source GeoPackage



# Read polygons from GeoPackages
reference_gdf = gpd.read_file(reference_gpkg)
source_gdf = gpd.read_file(source_gpkg)

# Assuming single polygon per GeoPackage for simplicity
reference_polygon = reference_gdf.geometry.unary_union
source_polygon = source_gdf.geometry.unary_union

# Calculate geometric differences and intersection
intersection = reference_polygon.intersection(source_polygon)
difference_ref = reference_polygon.difference(source_polygon)
difference_src = source_polygon.difference(reference_polygon)

# Quantitative Metrics
intersection_area = intersection.area
difference_area_ref = difference_ref.area
difference_area_src = difference_src.area

print(f"Intersection Area (Overlap): {intersection_area}")
print(f"Difference Area (Reference not in Source): {difference_area_ref}")
print(f"Difference Area (Source not in Reference): {difference_area_src}")

# Create GeoDataFrame to save differences
diff_gdf = gpd.GeoDataFrame({
    'geometry': [difference_ref, difference_src],
    'type': ['Reference not in Source', 'Source not in Reference']
}, crs=reference_gdf.crs)

# Output GeoPackage for differences
output_gpkg = "polygon_differences.gpkg"  # Replace with your desired output path
diff_gdf.to_file(output_gpkg, layer='differences', driver="GPKG")

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the reference and source polygons
reference_gdf.boundary.plot(ax=ax, color='black', linestyle='--', linewidth=1, label='Reference')
source_gdf.boundary.plot(ax=ax, color='black', linestyle='-', linewidth=1, label='Source')

# Plot intersection, differences
if not intersection.is_empty:
    gpd.GeoSeries(intersection).plot(ax=ax, color='green', alpha=0.5, label='Overlap')

if not difference_ref.is_empty:
    gpd.GeoSeries(difference_ref).plot(ax=ax, color='red', alpha=0.6, label='Reference not in Source')

if not difference_src.is_empty:
    gpd.GeoSeries(difference_src).plot(ax=ax, color='blue', alpha=0.6, label='Source not in Reference')

# Adding legends and titles
ax.legend()
plt.title("Polygon Differences Visualization")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.grid(True)
plt.show()
