import geopandas as gpd
import fiona
import pandas as pd

# Paths to the geopackages
path1 = r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\Centerlines_LSDTopo\Centerlines\ME_clipped.gpkg"
path2 = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\ME_centerline.gpkg"

# List layers in each geopackage
print("Layers in the first geopackage:")
layers1 = fiona.listlayers(path1)
print(layers1)

print("\nLayers in the second geopackage:")
layers2 = fiona.listlayers(path2)
print(layers2)

# Assuming the layers you want to compare are the first ones
layer1 = layers1[0]
layer2 = layers2[0]

# Read the specified layers
gdf1 = gpd.read_file(path1, layer=layer1)
gdf2 = gpd.read_file(path2, layer=layer2)

# Ensure both GeoDataFrames use the same CRS
if gdf1.crs != gdf2.crs:
    print("\nCoordinate Reference Systems are different. Reprojecting the second dataset to match the first.")
    gdf2 = gdf2.to_crs(gdf1.crs)
else:
    print("\nCoordinate Reference Systems are identical.")

# Print basic properties
print("\n--- Basic Properties ---")
print(f"CRS of the datasets: {gdf1.crs}")
print(f"Number of features in the first dataset: {len(gdf1)}")
print(f"Number of features in the second dataset: {len(gdf2)}")
print(f"Geometry types in the first dataset: {gdf1.geom_type.unique()}")
print(f"Geometry types in the second dataset: {gdf2.geom_type.unique()}")

# Compare attribute columns
print("\n--- Comparing Attribute Columns ---")
columns1 = set(gdf1.columns)
columns2 = set(gdf2.columns)
common_columns = columns1.intersection(columns2)
print(f"Columns in the first dataset: {columns1}")
print(f"Columns in the second dataset: {columns2}")
print(f"Common columns: {common_columns}")

# Check for differences in attributes
print("\n--- Comparing Attributes ---")
for col in common_columns:
    if col != 'geometry':
        if gdf1[col].equals(gdf2[col]):
            print(f"Attribute '{col}' is identical in both datasets.")
        else:
            print(f"Attribute '{col}' differs between datasets.")

# Compare geometries
print("\n--- Comparing Geometries ---")
geometry_equal = gdf1.geometry.equals(gdf2.geometry)


print("Geometries differ between the datasets.")
differing_indices = gdf1.index[gdf1.geometry != gdf2.geometry]
print(f"Indices with differing geometries: {differing_indices}")
# Detailed comparison for differing geometries
for idx in differing_indices:
    geom1 = gdf1.geometry.iloc[idx]
    geom2 = gdf2.geometry.iloc[idx]
    length1 = geom1.length
    length2 = geom2.length
    print(f"\n--- Difference at Index {idx} ---")
    print(f"Length in first dataset: {length1}")
    print(f"Length in second dataset: {length2}")
    print(f"Length difference: {abs(length1 - length2)}")

# Compare total lengths
total_length1 = gdf1.geometry.length.sum()
total_length2 = gdf2.geometry.length.sum()
print("\n--- Total Lengths ---")
print(f"Total length in the first dataset: {total_length1}")
print(f"Total length in the second dataset: {total_length2}")
print(f"Total length difference: {abs(total_length1 - total_length2)}")

# Compare bounding boxes
print("\n--- Bounding Boxes ---")
bbox1 = gdf1.total_bounds
bbox2 = gdf2.total_bounds
print(f"Bounding box of the first dataset: {bbox1}")
print(f"Bounding box of the second dataset: {bbox2}")
if not (bbox1 == bbox2).all():
    print("Bounding boxes differ between datasets.")

# Optional: Visual comparison (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    print("\n--- Visual Comparison ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf1.plot(ax=ax, color='blue', label='First Dataset')
    gdf2.plot(ax=ax, color='red', linestyle='--', label='Second Dataset')
    plt.legend()
    plt.title('Geometries of the Two Datasets')
    plt.show()
except ImportError:
    print("Matplotlib is not installed. Skipping visual comparison.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")
