from smooth_stream import interpolate_geopackage
from get_perpendiculars import create_smooth_perpendicular_lines
import os

# Define parameters
input_valley_centerline_list = [
 r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\UW_centerline.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\ME_centerline.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\MM_centerline.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\MW_centerline.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\UE_centerline.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\UM_centerline.gpkg",   
]
#"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\UM2_centerline.gpkg"

output_directory = r"Y:\ATD\GIS\Bennett\Valley Geometry\Perpendiculars"

for input_valley_centerline in input_valley_centerline_list:
    watershed = os.path.basename(input_valley_centerline).split("_")[0]
    smoothing_interval = 100  # in CRS units
    max_valley_width = 75  # in CRS units
    segment_spacing = 10  # in CRS units

    output_smoothed_valley = os.path.join(os.path.dirname(input_valley_centerline), f"_CL_smooth.gpkg")
    output_perpendiculars = os.path.join(output_directory, f"UM2_smooth_perpendiculars_{segment_spacing}m.gpkg")

    # Call the function
    interpolate_geopackage(
        input_gpkg=input_valley_centerline,
        output_gpkg=output_smoothed_valley,
        interval=smoothing_interval
    )


    perp_lines = create_smooth_perpendicular_lines(output_smoothed_valley, line_length = max_valley_width*1.5, 
                                                spacing=segment_spacing, window=max_valley_width*1.5, 
                                                output_path=output_perpendiculars)
    perp_lines.to_file(output_perpendiculars, driver='GPKG')
        
