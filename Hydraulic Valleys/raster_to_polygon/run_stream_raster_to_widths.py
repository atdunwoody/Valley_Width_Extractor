# run_processing.py
import os
from stream_raster_to_widths import process_rasters

def main():
    # Define your input information here

    raster_list = [
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_0o5cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_0o25cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_1cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_2cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_3cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_4cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_5cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_6cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_7cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_8cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_9cms.tif",
        "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Rasters/ME/ME_10cms.tif",
    ]

    # Path to the perpendicular lines GeoPackage
    lines_gpkg_path = "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Perpendiculars/ME_perps_5m.gpkg"

    # Define the output directory where GPKG files will be saved
    output_dir = "Y:/ATD/GIS/Bennett/Valley Widths/Valley_Footprints/Hydraulic Model/Max Depth Lines/ME_5m"
    
    # Path to save the connected polygon GeoPackage
    polygon_output_gpkg = os.path.join(output_dir, "connected_polygon.gpkg")

    # Call the processing function
    process_rasters(
        raster_list=raster_list,
        lines_gpkg_path=lines_gpkg_path,
        output_dir=output_dir,
        polygon_output_gpkg=polygon_output_gpkg
    )

if __name__ == "__main__":
    main()
