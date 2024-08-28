# Valley_Width_Extractor
 
1. Get stream centerlines from LSDTopoTools. These will come as a geojson containing points with stream order and other attributes.  
2. Run get_streams_from_LSDTopo.py to get a pruned stream network from LSDTopoTools channel network output
3. Merge these single line segements into a single line by running merge_multipart_lines.py
4. Get perpendiculars by running get_perpendiculars.py
5. 