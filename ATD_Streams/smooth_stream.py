import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import argparse
import sys
import fiona
from typing import Optional

def interpolate_linestring(line: LineString, interval: float) -> LineString:
    """
    Interpolate points along the LineString at regular intervals and create a new LineString.

    Parameters:
    - line: shapely.geometry.LineString
    - interval: float, distance between interpolated points

    Returns:
    - shapely.geometry.LineString
    """
    if not isinstance(line, LineString):
        raise TypeError("Input geometry must be a LineString.")

    # Total length of the LineString
    total_length = line.length

    if interval <= 0:
        raise ValueError("Interval must be a positive number.")

    # Generate distances at which to interpolate
    distances = [i * interval for i in range(int(total_length // interval) + 1)]

    # Ensure the last point is included
    if distances[-1] < total_length:
        distances.append(total_length)

    # Interpolate points
    interpolated_points = [line.interpolate(distance) for distance in distances]

    # Create a new LineString from interpolated points
    new_line = LineString(interpolated_points)
    return new_line

def process_geometry(geometry, interval):
    """
    Process a single geometry, handling LineString and MultiLineString.

    Parameters:
    - geometry: shapely.geometry.LineString or MultiLineString
    - interval: float, interpolation interval

    Returns:
    - shapely.geometry.LineString or MultiLineString
    """
    if isinstance(geometry, LineString):
        return interpolate_linestring(geometry, interval)
    elif isinstance(geometry, MultiLineString):
        # Flatten all LineStrings in the MultiLineString into a single LineString
        flattened_coords = []
        for line in geometry.geoms:
            flattened_coords.extend(line.coords)
        combined_line = LineString(flattened_coords)
        return interpolate_linestring(combined_line, interval)
    else:
        raise TypeError("Geometry must be a LineString or MultiLineString.")

def interpolate_geopackage(
    input_gpkg: str,
    output_gpkg: str,
    interval: float,
    layer_name: Optional[str] = None
) -> None:
    """
    Process the input GeoPackage to create smoothed LineStrings and save to output GeoPackage.

    Parameters:
    - input_gpkg: str, path to input GeoPackage
    - output_gpkg: str, path to output GeoPackage
    - interval: float, interpolation interval
    - layer_name: Optional[str], name of the layer containing the LineStrings
    """
    # Determine available layers
    try:
        available_layers = fiona.listlayers(input_gpkg)
    except Exception as e:
        print(f"Error accessing input GeoPackage: {e}")
        sys.exit(1)

    if not available_layers:
        print("No layers found in the input GeoPackage.")
        sys.exit(1)

    # If no layer name is provided, use the first layer
    if layer_name is None:
        layer_name = available_layers[0]
        print(f"No layer name provided. Using the first layer: '{layer_name}'")
    elif layer_name not in available_layers:
        print(f"Layer '{layer_name}' not found in the input GeoPackage.")
        print(f"Available layers: {available_layers}")
        sys.exit(1)

    # Read the specified layer
    try:
        gdf = gpd.read_file(input_gpkg, layer=layer_name)
    except Exception as e:
        print(f"Error reading layer '{layer_name}' from input GeoPackage: {e}")
        sys.exit(1)

    if gdf.empty:
        print(f"Layer '{layer_name}' is empty.")
        sys.exit(1)

    # Check for LineString or MultiLineString geometries
    valid_geom_types = ['LineString', 'MultiLineString']
    if not all(gdf.geometry.type.isin(valid_geom_types)):
        print(f"Layer '{layer_name}' does not contain only LineString or MultiLineString geometries.")
        sys.exit(1)

    # Process each geometry
    smoothed_geometries = []
    for idx, geom in enumerate(gdf.geometry):
        try:
            smoothed_geom = process_geometry(geom, interval)
            smoothed_geometries.append(smoothed_geom)
        except Exception as e:
            print(f"Error processing geometry at index {idx}: {e}")
            sys.exit(1)

    # Create a new GeoDataFrame with smoothed geometries
    smoothed_gdf = gdf.copy()
    smoothed_gdf['geometry'] = smoothed_geometries

    # Save to the output GeoPackage
    try:
        # Check if output GeoPackage already exists
        if fiona.supported_drivers.get("GPKG"):
            import os
            os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)
            smoothed_gdf.to_file(output_gpkg, driver="GPKG")
        else:
            print("GeoPackage driver not supported.")
            sys.exit(1)
        print(f"Smoothed LineStrings saved to '{output_gpkg}' in layer '{layer_name}'.")
    except Exception as e:
        print(f"Error writing to output GeoPackage: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a less sinuous (smoothed) version of stream LineStrings in a GeoPackage "
            "by interpolating points at a specified interval."
        )
    )
    parser.add_argument("input_gpkg", help="Path to the input GeoPackage (.gpkg)")
    parser.add_argument("output_gpkg", help="Path for the output GeoPackage (.gpkg)")
    parser.add_argument("interval", type=float, help="Interpolation interval (in CRS units, e.g., meters)")
    parser.add_argument(
        "--layer",
        "-l",
        type=str,
        default=None,
        help="Name of the layer containing the LineStrings (optional). If not provided, the first layer is used."
    )

    args = parser.parse_args()

    interpolate_geopackage(
        input_gpkg=args.input_gpkg,
        output_gpkg=args.output_gpkg,
        interval=args.interval,
        layer_name=args.layer
    )

if __name__ == "__main__":
    main()
