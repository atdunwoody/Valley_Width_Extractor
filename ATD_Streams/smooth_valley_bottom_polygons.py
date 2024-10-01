import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import os

def smooth_polygon(input_gpkg, output_gpkg, smoothing_window=5):
    """
    Smooths polygons from an input GeoPackage by reducing the variance in their exterior boundaries.
    
    Parameters:
    - input_gpkg (str): Path to the input GeoPackage containing the polygons.
    - output_gpkg (str): Path where the smoothed GeoPackage will be saved.
    - smoothing_window (int): Window size for the moving average. Must be an odd integer >= 3.
    
    Returns:
    - None
    """
    
    if smoothing_window < 3 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd integer >= 3.")
    
    def moving_average(coords, window):
        """
        Applies a moving average to the coordinates without assuming circularity.
        
        Parameters:
        - coords (np.ndarray): Array of shape (N, 2) representing the exterior coordinates.
        - window (int): Window size for the moving average.
        
        Returns:
        - smoothed_coords (np.ndarray): Array of smoothed coordinates.
        """
        # Ensure coordinates are in a NumPy array
        coords = np.array(coords)
        
        # Separate x and y
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Number of points
        N = len(x)
        
        # Pad the coordinates at the start and end to handle boundaries
        pad_size = window // 2
        x_padded = np.pad(x, (pad_size, pad_size), mode='edge')
        y_padded = np.pad(y, (pad_size, pad_size), mode='edge')
        
        # Initialize smoothed arrays
        x_smooth = np.zeros(N)
        y_smooth = np.zeros(N)
        
        # Apply moving average
        for i in range(N):
            x_smooth[i] = np.mean(x_padded[i:i + window])
            y_smooth[i] = np.mean(y_padded[i:i + window])
        
        # Combine back
        smoothed = np.vstack((x_smooth, y_smooth)).T
        
        return smoothed

    def smooth_geometry(geom, window):
        """
        Smooths the exterior of a polygon or multipolygon.
        
        Parameters:
        - geom (shapely.geometry): The geometry to smooth.
        - window (int): Window size for the moving average.
        
        Returns:
        - smoothed_geom (shapely.geometry): The smoothed geometry.
        """
        if isinstance(geom, Polygon):
            exterior = list(geom.exterior.coords)
            # Remove the closing point for smoothing
            exterior_unique = exterior[:-1]
            smoothed_exterior = moving_average(exterior_unique, window)
            # Ensure the polygon is closed
            smoothed_exterior = np.vstack((smoothed_exterior, smoothed_exterior[0]))
            
            # Handle interiors (holes) similarly if needed
            interiors = []
            for interior in geom.interiors:
                interior_coords = list(interior.coords)
                interior_unique = interior_coords[:-1]
                smoothed_interior = moving_average(interior_unique, window)
                smoothed_interior = np.vstack((smoothed_interior, smoothed_interior[0]))
                interiors.append(smoothed_interior)
            
            return Polygon(smoothed_exterior, interiors)
        
        elif isinstance(geom, MultiPolygon):
            smoothed_polygons = [smooth_geometry(part, window) for part in geom]
            return MultiPolygon(smoothed_polygons)
        
        else:
            # If not a Polygon or MultiPolygon, return as is
            return geom

    # Check if input file exists
    if not os.path.exists(input_gpkg):
        raise FileNotFoundError(f"Input GeoPackage not found: {input_gpkg}")
    
    # Read the input GeoPackage
    gdf = gpd.read_file(input_gpkg)
    
    # Apply make_valid to ensure all geometries are valid
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    
    # Smooth each geometry
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: smooth_geometry(geom, smoothing_window))
    
    # Optionally, you can set a coordinate reference system (CRS) if needed
    # gdf.set_crs(epsg=XXXX, inplace=True)
    
    # Write the smoothed geometries to the output GeoPackage
    gdf.to_file(output_gpkg, driver="GPKG")
    
    print(f"Smoothed GeoPackage saved to: {output_gpkg}")

# Example usage
if __name__ == "__main__":
    valley_bottom_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\UE\20240930_16h36m\UE_clipped_valleys_wavelets.gpkg"
    output_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\UE\20240930_16h36m\UE_clipped_valleys_wavelets_smoothed_10.gpkg"
    
    smooth_polygon(valley_bottom_gpkg, output_gpkg, smoothing_window=11)
