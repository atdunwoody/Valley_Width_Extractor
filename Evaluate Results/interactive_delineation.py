import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point
import numpy as np
import json
import os

# Function to sample elevation along a line
def sample_elevation_along_line(line, dem_path, num_points=100):
    with rasterio.open(dem_path) as src:
        # Generate equally spaced points along the line
        distances = np.linspace(0, line.length, num_points)
        points = [line.interpolate(distance) for distance in distances]
        # Sample elevation at each point
        elevations = []
        for point in points:
            # rasterio.sample returns a generator of arrays
            sample = next(src.sample([(point.x, point.y)]))
            elevations.append(sample[0])
        elevations = np.array(elevations)
        return distances, elevations, points  # Returning the points in geometry coordinates

# Modified Function to find all intersection points between the selected elevation and terrain profile
def find_intersections(elevations, distances, points, selected_elevation):
    intersections = []
    for i in range(len(elevations) - 1):
        y0 = elevations[i]
        y1 = elevations[i + 1]
        # Check if the selected elevation crosses between y0 and y1
        if (y0 - selected_elevation) * (y1 - selected_elevation) < 0:
            # Perform linear interpolation to find the exact distance
            alpha = (selected_elevation - y0) / (y1 - y0)
            distance = distances[i] + alpha * (distances[i + 1] - distances[i])
            # Interpolate coordinates
            x = points[i].x + alpha * (points[i + 1].x - points[i].x)
            y_coord = points[i].y + alpha * (points[i + 1].y - points[i].y)
            intersections.append({
                "distance": float(distance),
                "coordinates": (float(x), float(y_coord))
            })
        elif y0 == selected_elevation and y1 != selected_elevation:
            # Exact match at the start point
            intersections.append({
                "distance": float(distances[i]),
                "coordinates": (float(points[i].x), float(points[i].y))
            })
        elif y1 == selected_elevation and y0 != selected_elevation:
            # Exact match at the end point
            intersections.append({
                "distance": float(distances[i + 1]),
                "coordinates": (float(points[i + 1].x), float(points[i + 1].y))
            })
    return intersections

# Callback function for updating the position of the horizontal line and showing the depth
def on_move(event, line, ax, min_elevation, depth_annotation):
    if event.inaxes == ax:  # Make sure the mouse is in the correct plot
        y = event.ydata
        if y is not None:
            line.set_ydata([y, y])  # Update the y-position of the line
            current_depth = float(y - min_elevation)  # Ensure current_depth is a scalar value
            depth_annotation.set_text(f'Depth: {current_depth:.2f} m')
            plt.draw()  # Redraw the plot with the updated line and depth

# Callback function for double-click to confirm the position of the horizontal line and save the plot
def on_double_click(event, line, results, elevations, distances, points, fig, filename, ax):
    if event.dblclick:
        # Record the y-position of the line (elevation)
        final_y_position = float(line.get_ydata()[0])  # Convert to a scalar
        
        print(f"Selected Elevation: {final_y_position}")
        
        # Find all intersections between the red line and the terrain
        intersections = find_intersections(elevations, distances, points, final_y_position)
        
        # Store the result as a dictionary
        results.append({
            "selected_elevation": float(final_y_position),
            "intersections": intersections
        })
        
        # Plot markers for intersections on the elevation profile
        for intersection in intersections:
            distance = intersection["distance"]
            elevation = final_y_position
            #ax.plot(distance, elevation, 'bo')  # Mark intersection points with blue circles
            #ax.axvline(x=distance, color='blue', linestyle='--', alpha=0.7)  # Optional: Draw vertical dashed lines
        
        # Update the plot to show the markers
        fig.savefig(filename)
        print(f"Plot saved to {filename}")
        
        plt.close()  # Close the current plot to move to the next one

# Function to plot profiles with an interactive horizontal line and depth display
def plot_profiles_with_drag(lines_gdf, dem_path, output_folder):
    results = {}
    all_points = []  # To store intersection points for saving to GeoPackage
    for idx, row in lines_gdf.iterrows():
        line = row.geometry
        distances, elevations, points = sample_elevation_along_line(line, dem_path)
        min_elevation = float(np.min(elevations))  # Get the minimum elevation in the profile

        # Plot the elevation profile
        fig, ax = plt.subplots()
        ax.plot(distances, elevations, label='Elevation Profile')
        ax.set_title(f"Line {idx + 1}")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")

        # Add a draggable horizontal red line
        init_y = (float(np.max(elevations)) + min_elevation) / 2  # Initial y-position of the line
        horizontal_line = Line2D([distances[0], distances[-1]], [init_y, init_y], color='red', lw=0.5)
        ax.add_line(horizontal_line)

        # Add a depth annotation
        depth_annotation = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')

        # List to store the selected elevation and its intersections
        clicked_elevation = []

        # Define the output filename for the plot
        output_plot = os.path.join(output_folder, "Plots" f"line_{idx + 1}.png")
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)  
        # Connect the event handlers for mouse movement and double-click
        cid_move = fig.canvas.mpl_connect(
            'motion_notify_event', 
            lambda event, hl=horizontal_line, a=ax, me=min_elevation, da=depth_annotation: on_move(event, hl, a, me, da)
        )
        cid_click = fig.canvas.mpl_connect(
            'button_press_event', 
            lambda event, ln=horizontal_line, res=clicked_elevation, elev=elevations, dist=distances, pts=points, fig=fig, filename=output_plot, ax=ax: on_double_click(event, ln, res, elev, dist, pts, fig, filename, ax)
        )

        plt.legend()
        plt.show()

        if clicked_elevation:
            selected_elev = clicked_elevation[0]["selected_elevation"]
            intersections = clicked_elevation[0]["intersections"]
            results[f'line_{idx + 1}'] = {
                "selected_elevation": selected_elev,
                "intersections": intersections,
                "elevations": elevations.tolist(),  # Ensure this is a list
                "distances": distances.tolist()      # Ensure this is a list
            }

            # Collect intersection points and their elevations for GeoPackage
            for intersection in intersections:
                x, y_coord = intersection["coordinates"]
                all_points.append({
                    "geometry": Point(x, y_coord),
                    "line_id": idx + 1,  # Assuming line IDs start at 1
                    "selected_elevation": selected_elev,
                    "distance": intersection["distance"]
                })
        else:
            results[f'line_{idx + 1}'] = {
                "selected_elevation": None,
                "intersections": [],
                "elevations": elevations.tolist(),
                "distances": distances.tolist()
            }

    return results, all_points

# Function to save results to a GeoPackage
def save_points_to_gpkg(all_points, crs, output_gpkg):
    if all_points:
        points_gdf = gpd.GeoDataFrame(all_points, crs=crs)
        points_gdf.to_file(output_gpkg, layer='elevation_intersections', driver='GPKG')
        print(f"Intersection points saved to {output_gpkg}")
    else:
        print("No intersection points to save.")

# Function to save results to a JSON file
def save_results_to_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

# Main function
def main(gpkg_path, dem_path, output_json, output_gpkg, output_folder):
    # Load the lines from the geopackage
    lines_gdf = gpd.read_file(gpkg_path)

    # Plot profiles and record elevations based on user interaction
    results, all_points = plot_profiles_with_drag(lines_gdf, dem_path, output_folder)

    # Save the intersection points to GeoPackage
    if not lines_gdf.empty:
        save_points_to_gpkg(all_points, lines_gdf.crs, output_gpkg)

    # Save the results to a JSON file
    save_results_to_json(results, output_json)

if __name__ == "__main__":
    gpkg_path = r"Y:\ATD\GIS\Bennett\Valley Geometry\Perpendiculars\ME_smooth_perpendiculars_200m.gpkg"
    dem_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\filled_dem.tif"
    output_folder = r"Y:\ATD\GIS\Bennett\Valley Bottoms\Valley_Footprints\Manual Delineation\ME"
    
    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, "manual_delineation.json")
    output_gpkg = os.path.join(output_folder, "manual_delineation.gpkg")
    main(gpkg_path, dem_path, output_json, output_gpkg, output_folder)
