import os
import shutil
from utils.convert_to_ENVI_bil import convert_dem_to_envi_bil
# Define the paths
tif_folder = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\Watershed_Clipped"
param_file_path = r"C:\LSDTopoTools\2021_LIDAR.param"
destination_root = r"C:\LSDTopoTools"  # Root folder for new folders

# Ensure the destination root folder exists
os.makedirs(destination_root, exist_ok=True)

# Iterate through all the .tif files in the folder
for filename in os.listdir(tif_folder):
    if filename.endswith(".tif"):
        # Full path to the .tif file
        tif_file_path = os.path.join(tif_folder, filename)
        
        # Create a new folder named after the .tif file (without extension)
        folder_name = os.path.splitext(filename)[0]
        new_folder_path = os.path.join(destination_root, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Copy the .tif file to the new folder
        shutil.copy2(tif_file_path, new_folder_path)
        
        # Copy the .param file to the new folder
        new_param_file_path = os.path.join(new_folder_path, os.path.basename(param_file_path))
        shutil.copy2(param_file_path, new_param_file_path)
        
        # Modify the .param file to update the "read fname:" line
        with open(new_param_file_path, 'r') as param_file:
            param_contents = param_file.readlines()
        
        with open(new_param_file_path, 'w') as param_file:
            for line in param_contents:
                if line.startswith("read fname:"):
                    line = f"read fname: {filename}\n"
                param_file.write(line)
        convert_dem_to_envi_bil(tif_file_path, os.path.join(new_folder_path, folder_name + ".bil"))
print("Folders created and files copied successfully.")
