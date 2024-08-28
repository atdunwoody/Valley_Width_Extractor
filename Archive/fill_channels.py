import rasterio
import geopandas as gpd
from rasterio.mask import mask
import numpy as np
from scipy.ndimage import generic_filter

def fill_channel_with_nodata(raster_path, channel_gpkg, output_raster_path, buffer_size = 0.0001):
    # Load the channel polygon from the GeoPackage
    channels = gpd.read_file(channel_gpkg)
    
    #buffer the channel polygon by 1 meter to ensure the raster is fully masked
    channels['geometry'] = channels.buffer(buffer_size)
    
    # Open the elevation raster
    with rasterio.open(raster_path) as src:
        # Check if CRS of the raster and vector layers match, reproject vector if necessary
        if src.crs != channels.crs:
            print("Reprojecting vector layer to match raster CRS...")
            channels = channels.to_crs(src.crs)
        
        # Mask the raster with the channel polygon, setting masked areas to NoData
        out_image, out_transform = mask(src, channels.geometry, invert=True, nodata=src.nodata)

        # Combine the mask with the original data, filling the masked areas with NoData
        filled_elevation_data = np.where(np.isnan(out_image[0]), src.nodata, out_image[0])

        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": filled_elevation_data.shape[0],
            "width": filled_elevation_data.shape[1],
            "transform": out_transform,
            "nodata": src.nodata
        })

        # Write the modified raster to a new file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(filled_elevation_data, 1)



def fill_channel_with_neighborhood_avg(raster_path, channel_gpkg, output_raster_path, buffer_size=0.0001, window_size=3):
    # Load the channel polygon from the GeoPackage
    channels = gpd.read_file(channel_gpkg)
    
    # Buffer the channel polygon to ensure the raster is fully masked
    channels['geometry'] = channels.buffer(buffer_size)
    
    # Open the elevation raster
    with rasterio.open(raster_path) as src:
        # Check if CRS of the raster and vector layers match, reproject vector if necessary
        if src.crs != channels.crs:
            print("Reprojecting vector layer to match raster CRS...")
            channels = channels.to_crs(src.crs)
        
        # Mask the raster with the channel polygon
        out_image, out_transform = mask(src, channels.geometry, invert=True, nodata=src.nodata)
        
        # Define a function to compute the neighborhood average, ignoring NaN values
        def neighborhood_mean(window):
            return np.nanmean(window)
        
        # Apply the neighborhood mean filter to the masked image
        # The size of the window is defined by the `window_size` parameter
        filled_elevation_data = generic_filter(
            out_image[0], 
            function=neighborhood_mean, 
            size=window_size, 
            mode='nearest'
        )
        
        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": filled_elevation_data.shape[0],
            "width": filled_elevation_data.shape[1],
            "transform": out_transform,
            "nodata": src.nodata
        })

        # Write the modified raster to a new file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(filled_elevation_data, 1)

raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_ME_clip_filtered.tif"
channel_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\ATD_algorithm\Channel_filt_2dir\Valley_Footprint.gpkg"
output_raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_ME_clip_filtered_chan_fill.tif"


#fill_channel_with_nodata(raster_path, channel_gpkg, output_raster_path, buffer_size = 10)

fill_channel_with_neighborhood_avg(raster_path, channel_gpkg, output_raster_path, buffer_size=5, window_size=3)

