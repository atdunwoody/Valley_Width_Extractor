import os
from osgeo import gdal

def warp_to_template(input_folder, template_raster, output_folder):
    """
    Warps all .tif files in the input folder to match the template raster's spatial reference, resolution, and extent.
    
    Parameters:
    - input_folder: Path to the folder containing input .tif files.
    - template_raster: Path to the template raster that the input rasters will be warped to match.
    - output_folder: Path to the folder where the warped .tif files will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the template raster
    template_ds = gdal.Open(template_raster)
    if template_ds is None:
        raise FileNotFoundError(f"Template raster not found: {template_raster}")

    # Get the geotransform and projection from the template raster
    template_proj = template_ds.GetProjection()
    template_geotransform = template_ds.GetGeoTransform()
    template_width = template_ds.RasterXSize
    template_height = template_ds.RasterYSize
    
    # Loop through all .tif files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tif'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # Open the input raster
            input_ds = gdal.Open(input_path)
            if input_ds is None:
                print(f"Failed to open input raster: {input_path}")
                continue

            # Warp the input raster to match the template
            warp_options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(template_geotransform[0], 
                              template_geotransform[3] + template_height * template_geotransform[5],
                              template_geotransform[0] + template_width * template_geotransform[1], 
                              template_geotransform[3]),
                width=template_width,
                height=template_height,
                dstSRS=template_proj,
                resampleAlg='bilinear'
            )

            gdal.Warp(output_path, input_ds, options=warp_options)

            # Clean up
            input_ds = None
            print(f"Warped {input_path} to {output_path}")

    # Close template dataset
    template_ds = None

input_folder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\MM"
template_raster = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_bennett_clip.tif"
output_folder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\MM_warped"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
warp_to_template(input_folder, template_raster, output_folder)
