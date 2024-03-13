
# NaviSense: Navigation Assistance for the Visually Impaired
## This project utilizes a combination of depth and color data from an RGB-D camera to create a navigation assistance system for visually impaired people.

<div style="text-align: center;">
    <img width="100%" src="/Navisense_demo.gif">
</div>

## About the Project
NaviSense employs two distinct methods for obstacle detection:

1. Floor Level Detection: A machine learning model known as PIDNet is used to identify ground-level obstacles like curbs or uneven sidewalks. This model was trained on a comprehensive dataset of images specifically captured around Chulalongkorn University.

2. Upper Level Detection: For obstacles situated higher up, such as hanging signs or doorways, NaviSense leverages the Open3D library. This library analyzes the 3D point cloud data obtained from the camera's depth sensor to recognize objects and establish boundaries around them.
For further details regarding the system's architecture and operation, please refer to the report or flowchart located within the project's documentation folder (docs).

## System Requirements
While NaviSense was developed and tested on a specific computer setup, it has the potential to run on other machines with comparable capabilities. Here's a breakdown of the requirements:

## Hardware:
A computer equipped with an Nvidia graphics card containing CUDA cores and sufficient VRAM (minimum recommendation: 6GB)
An Intel RealSense depth camera (specifically, the D455 model was used for development)
## Software:
Python with PyTorch (GPU-enabled version)
RealSense SDK
## Running the Code
1. **Install Dependencies:** Open a terminal window and navigate to the project directory (Navisense). Subsequently, run the following command to install the necessary Python libraries:
`pip install -r requirements.txt`

2. **GPU and Camera Verification:**
Ensure your graphics card is Nvidia and has CUDA cores enabled.
Utilize the RealSense SDK tools to confirm that your camera is recognized and functioning properly.

3. **Choosing Your Input Method:**
   - Recorded Data: If you possess a pre-recorded video file containing depth and color data (RGBD), you can utilize the recorded.py script. Simply modify the file path on line 29 of this script to point to your desired video. Then, execute the script from the terminal:

     `py ./scripts/recorded.py`

   - **Live Camera:** To employ NaviSense with a live camera feed, you have two options:

     - **Basic Window:** Run the realtime.py script for a basic OpenCV window display:

       `py ./scripts/realtime.py`
       
     - **Graphical User Interface (GUI):** For a more user-friendly interface, run the with_gui.py script:

       `py ./scripts/with_gui.py`
       
       Please note: The GUI window size might not automatically adjust to your screen size.

## Troubleshooting Tips
**Python Interpreter:** If "py" is not your designated Python interpreter command, replace it with the appropriate command for your system (e.g., "python3").
Terminal Directory: Before running any scripts, ensure your terminal is pointing to the main project directory (Navisense).

**Camera Connection Issues:** If you encounter problems connecting the camera, utilize the RealSense SDK tools to diagnose the issue.

**Changing Camera Model:** To utilize a different Intel RealSense camera model, adjust the settings within the rs_config.json file located in the scripts folder.

## Folder Description
**Navisense:** Contains the core functions necessary to run the code.

**PIDNet:** Includes a pre-trained PIDNet model with unnecessary files and scripts removed.

**docs:** Stores flowcharts that illustrate the system's architecture.

**scripts:** Holds the Python scripts used to operate the system.




