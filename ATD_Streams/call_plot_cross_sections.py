# call_plot_cross_sections.py

import logging
from plot_cross_sections import main
import os
    
def run_cross_section_plotting():
    # Define paths to input files
    perpendiculars_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_perpendiculars_1000m.gpkg"
    dem_raster = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Terrain\WBT_Outputs\filled_dem.tif"
    json_file = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\ATD_Streams\Results_0.01_rolling_variance_detection_1000m_json\flow_depths.json"     # Replace with the actual path to your JSON file
    output_folder = os.path.join(os.path.dirname(json_file), "cross_sections")  # Replace with the actual path to your output folder


    os.makedirs(output_folder, exist_ok=True)
    # Call the main function from plot_cross_sections.py
    try:
        main(perpendiculars_path, dem_raster, json_file, output_folder)
        print("Cross-section plots have been generated successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    run_cross_section_plotting()
