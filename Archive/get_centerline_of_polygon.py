import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split
import numpy as np
import rasterio
from rasterio.features import rasterize
from skimage.morphology import skeletonize
import os

def create_centerline(polygon_file, output_CL_file):
    """Process each polygon in the GeoPackage to create centerlines and save them."""
    def polygon_to_raster(poly, cell_size=1):
        """Convert a polygon to a raster array."""
        bounds = poly.bounds
        width = int(np.ceil((bounds[2] - bounds[0]) / cell_size))
        height = int(np.ceil((bounds[3] - bounds[1]) / cell_size))
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], cell_size, cell_size)
        raster = rasterize([(poly, 1)], out_shape=(height, width), transform=transform)
        return raster, transform

    def raster_to_centerline(raster, transform):
        """Convert raster array to a centerline geometry."""
        skeleton = skeletonize(raster == 1)
        points = [Point(*rasterio.transform.xy(transform, row, col, offset='center'))
                for row in range(skeleton.shape[0]) for col in range(skeleton.shape[1]) if skeleton[row, col]]
        if points:
            line = LineString(points)
            return line
        return None

    def calc_centerline(polygon, cell_size=1):
        """Main function to create centerline from a polygon."""
        raster, transform = polygon_to_raster(polygon, cell_size)
        centerline = raster_to_centerline(raster, transform)
        return centerline
    

    gdf = gpd.read_file(polygon_file)
    gdf['centerline'] = gdf['geometry'].apply(lambda x: calc_centerline(x))

    # Remove entries where no centerline was found
    centerlines_gdf = gdf.dropna(subset=['centerline'])
    centerlines_gdf = centerlines_gdf.set_geometry('centerline', drop=True)  # Set 'centerline' as the geometry column and drop the old one
    centerlines_gdf.crs = gdf.crs  # Ensure CRS is preserved
    # Save to a new GeoPackage
    centerlines_gdf.to_file(output_CL_file, layer='centerlines', driver='GPKG')

    return centerlines_gdf

    
def main():

    
    centerline_dir = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Centerlines"
    input_dir = r"Y:\ATD\GIS\Bennett\Channel Polygons"

    output_segment_dir = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Perpendiculars"
    if not os.path.exists(output_segment_dir):
        os.makedirs(output_segment_dir)
    watersheds = ['LM2', 'LPM', 'MM', 'MPM', 'UM1', 'UM2']
    
    for watershed in watersheds:
        #search for right raster by matching the watershed name
        for file in os.listdir(input_dir):
            if watershed in file and file.endswith('.gpkg'):
                input_path = os.path.join(input_dir, file)
                print(f"Input: {input_path}")
                break

        centerline_path = os.path.join(centerline_dir, f'{watershed} centerline.gpkg')
        output_segment_path = os.path.join(output_segment_dir, f'{watershed}_channel_segmented.gpkg')
        print(f"Processing watershed: {watershed}")
        
        print(f"Centerline: {centerline_path}")
        print(f"Output: {output_segment_path}\n")
        create_centerline(input_path, centerline_path)

    
if __name__ == '__main__':
    main()