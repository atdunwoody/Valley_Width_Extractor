from smooth_stream import interpolate_geopackage
from get_perpendiculars import create_smooth_perpendicular_lines

# Define parameters
input_valley_centerline = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Stream\Valley_CL.gpkg"
output_smoothed_valley = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Stream\Valley_CL_smooth.gpkg"
output_perpendiculars = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Stream\Perpendiculars_10m.gpkg"
smoothing_interval = 100  # in CRS units
max_valley_width = 150  # in CRS units
segment_spacing = 10  # in CRS units

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
    
