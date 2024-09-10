from osgeo import gdal
import os


def convert_dem_to_envi_bil(input_dem_path, output_bil_path):
    # Open the input DEM
    dem_dataset = gdal.Open(input_dem_path, gdal.GA_ReadOnly)
    if dem_dataset is None:
        print("Failed to open the DEM file.")
        return
    
    # Define the driver for the ENVI .bil format
    driver = gdal.GetDriverByName('ENVI')
    if driver is None:
        print("ENVI driver is not available.")
        return
    
    # Create a copy of the dataset to convert it into ENVI format
    driver.CreateCopy(output_bil_path, dem_dataset, options=["INTERLEAVE=BIL"])
    
    output_dataset = gdal.Open(output_bil_path, gdal.GA_ReadOnly)
    if output_dataset is not None:
        print("Projection is:", output_dataset.GetProjection())
    output_dataset = None
    
    # Properly close the datasets to flush to disk
    dem_dataset = None
    print("Conversion completed. The output file is stored at:", output_bil_path)


def main(input_dem, output_bil = None):
    
    input_dem = r"C:\LSDTopoTools\Bennett_Full\dem_2021_bennett_clip.tif"
    #replace tif with bil
    if output_bil is None:
        output_bil = os.path.splitext(input_dem)[0] + ".bil"
    convert_dem_to_envi_bil(input_dem, output_bil)
if __name__ == '__main__':
    main()
