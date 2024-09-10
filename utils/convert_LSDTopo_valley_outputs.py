import os
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

# List of input directories
input_dirs = [
    r"C:\LSDTopoTools\ME_clipped",
    r"C:\LSDTopoTools\MM_clipped",
    r"C:\LSDTopoTools\MW_clipped",
    r"C:\LSDTopoTools\UE_clipped",
    r"C:\LSDTopoTools\UM_clipped",
    r"C:\LSDTopoTools\UW_clipped"
]

# Function to convert .bil to .tif and set NoData value
def convert_bil_to_tif(input_path, output_path):
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,  # Update dtype if needed
            nodata=0,
            driver='GTiff'
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i, resampling=Resampling.nearest)
                data[data == 0] = profile['nodata']  # Set NoData value
                data[data == 4] = profile['nodata']  # Uncomment to keep streams
                dst.write(data, i)

# Function to convert .tif to .gpkg with polygons
def convert_tif_to_gpkg(tif_path, gpkg_path):
    with rasterio.open(tif_path) as src:
        mask = src.dataset_mask()
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(src.read(1), mask=mask, transform=src.transform)
            )
        )

        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

        # Fill holes by buffering, uniting, and eroding
        if not gdf.empty:
            # Buffer by 2 meters to fill holes
            gdf['geometry'] = gdf.buffer(2)
            
            # Dissolve all polygons into a single shape
            unified = gdf.unary_union
            
            # Erode the shape by 2 meters (negative buffer)
            eroded = unified.buffer(-2)
            
            # Convert back to GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=[eroded], crs=src.crs)

        # Save to GeoPackage
        gdf.to_file(gpkg_path, driver="GPKG")

# Process each input directory and its subdirectories
for input_dir in input_dirs:
    watershed_name = os.path.basename(input_dir).split("_")[0]
    # Walk through subdirectories
    for subdir, _, files in os.walk(input_dir):
        input_file = os.path.join(subdir, "2021_LIDAR_modified_valley.bil")
        if os.path.exists(input_file):
            # Construct output filenames based on the subdirectory name
            subdir_name = os.path.basename(subdir).split("_")[1]
            output_tif = os.path.join(subdir, f"{subdir_name}_2021_LIDAR_valley.tif")
            output_gpkg = os.path.join(os.path.dirname(subdir), f"{watershed_name}_{subdir_name}_2021_LIDAR_valley.gpkg")
            print(f"Saving gpkg to {output_gpkg}")
            # Convert .bil to .tif
            convert_bil_to_tif(input_file, output_tif)
            print(f"Converted {input_file} to {output_tif}")
            
            # Convert .tif to .gpkg
            convert_tif_to_gpkg(output_tif, output_gpkg)
            print(f"Converted {output_tif} to {output_gpkg}")
        else:
            print(f"{input_file} does not exist.")

print("Conversion completed.")
