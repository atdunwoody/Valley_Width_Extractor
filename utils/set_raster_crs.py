import rasterio
from rasterio.crs import CRS
from rasterio.errors import CRSError
import sys
import os


def is_raster_file(filename):
    """
    Check if a file is a raster based on its extension.
    You can add more extensions if needed.
    """
    raster_extensions = ['.tif', '.tiff', '.img', '.vrt', '.kea', '.jp2']
    _, ext = os.path.splitext(filename)
    return ext.lower() in raster_extensions

def assign_crs_if_missing(input_path, output_path, new_crs):
    """
    Assigns a new CRS to the raster if it's missing and saves it to the output path.
    """
    try:
        with rasterio.open(input_path) as src:
            crs = src.crs
            if crs:
                print(f"CRS already set for '{os.path.basename(input_path)}': {crs}")
                return  # CRS exists; no action needed

            print(f"No CRS found for '{os.path.basename(input_path)}'. Assigning CRS '{new_crs}'.")

            # Read the metadata and update the CRS
            metadata = src.meta.copy()
            metadata['crs'] = new_crs

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write to the new raster file with updated CRS
            with rasterio.open(output_path, 'w', **metadata) as dst:
                for i in range(1, src.count + 1):
                    band = src.read(i)
                    dst.write(band, i)
        
        print(f"Assigned CRS '{new_crs}' to '{os.path.basename(output_path)}' and saved to '{output_path}'.\n")

    except CRSError:
        print(f"Error: '{new_crs}' is not a valid CRS.\n")
    except Exception as e:
        print(f"An error occurred while processing '{os.path.basename(input_path)}': {e}\n")

def process_rasters(input_dir, output_dir, default_crs):
    """
    Processes all raster files in the input directory.
    Assigns a default CRS to those without one and saves them to the output directory.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: The input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'.\n")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            sys.exit(1)

    raster_files = [f for f in os.listdir(input_dir) if is_raster_file(f)]

    if not raster_files:
        print(f"No raster files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(raster_files)} raster file(s) in '{input_dir}'.\n")

    for raster_file in raster_files:
        input_path = os.path.join(input_dir, raster_file)
        output_path = os.path.join(output_dir, raster_file)
        assign_crs_if_missing(input_path, output_path, default_crs)

def main():
    

    input_directory= r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point"
    output_directory = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point\CRS"
    os.makedirs(output_directory, exist_ok=True)
    # Define the CRS to assign if missing
    DEFAULT_CRS = 'EPSG:26913'

    process_rasters(input_directory, output_directory, DEFAULT_CRS)

if __name__ == "__main__":
    main()
