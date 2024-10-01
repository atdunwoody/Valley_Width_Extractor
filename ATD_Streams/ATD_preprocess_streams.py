from smooth_stream import interpolate_geopackage
from get_perpendiculars import create_smooth_perpendicular_lines
import os

# Define parameters
input_valley_centerline = r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LPM_centerline.gpkg"
output_directory = r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars"

smoothing_interval = 10  # in CRS units
max_valley_width = 75  # in CRS units
segment_spacing = 20  # in CRS units

output_smoothed_valley = os.path.join(output_directory, "LPM_CL_smooth.gpkg")
output_perpendiculars = os.path.join(output_directory, f"LPM_smooth_perpendiculars_{segment_spacing}m.gpkg")

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
    
