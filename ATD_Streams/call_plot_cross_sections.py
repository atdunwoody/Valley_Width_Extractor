# call_plot_cross_sections.py

import logging
from plot_cross_sections import main
import os
    
def run_cross_section_plotting(perpendiculars_path, dem_raster, json_file):
    # Define paths to input files
    output_folder = os.path.join(os.path.dirname(json_file), "cross_sections")  # Replace with the actual path to your output folder


    os.makedirs(output_folder, exist_ok=True)
    # Call the main function from plot_cross_sections.py
    try:
        main(perpendiculars_path, dem_raster, json_file, output_folder)
        print("Cross-section plots have been generated successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    perpendiculars_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\MW\20240930_17h37m\MW_clipped_perps_5m_hillslopes.gpkg"
    dem_raster = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\filled_dem.tif"
    json_file = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\MW\20240930_17h37m\flow_depths.json"  # Replace with the actual path to your JSON file

    run_cross_section_plotting(perpendiculars_path, dem_raster, json_file)
