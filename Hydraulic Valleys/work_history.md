## 9/24/2024 10:53AM

Added `get_valleys_from_HEC_polys_windowed.py` script to smooth out hydraulic model valley bottom outputs
It worked on smoothing them out. 

### Process
1. The xxth percentile (e.g. 95th percentile) of depths for each segment is for each flow raster
2. Flow raster depths are smoothed and a polyfit is performed
3. Inflection point is found based on change of sign of 2nd derivative of smoothed line
    - Note: Could also try using the polyfit line
4. Moving window average performed 5 segments ahead and behind current raster, and resulting depth is rounded to nearest depth raster value  
4. Depth raster associated with moving average inflection point is recorded
5. Rasters for each segment are stitched together  

### Future ideas for algorithm improvement: 
- Convert flow rasters to a polygon file
- Instead of rounding windowed average and inflection point to nearest flow raster value, get exact value
- Then interpolate width of valley bottom proportional to difference between depths
- e.g. if the inflection point is at 2.5 m depth, interpolate to width halfway between 2 and 3 m on the hydraulic model max depth raster 

## 9/24/2024 1:55 PM
Added `plot_terrain_vs_depth.py` to see visually how selected depths for each poylgon segment match up to inflection point graphs

## 9/24/2024 3:15 PM
Added `stream_raster_to_polygon.py`
    - Converts hydraulic model max depth rasters to poylgon outlines

Added `stream_raster_to_widths.py` 
    - Takes in perpendicular lines and max depth rasters
    - Outputs perpendicular lines clipped to the outline of the raster (from which widths can be extracted)