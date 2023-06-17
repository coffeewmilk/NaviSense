# NaviSense
Navigation assistance for visually impaired person project using the combination of depth and color data from the RGBD camera to navigate 

# About the Project
the system use two different method to detect obstacle
1. floor level obstacle detection is powered by PIDNet, an image segmentation machine learning model. The model is then trained with additional 300+ images of sidewalks around Chulalongkorn University.
2. upper floor level obstacle detection is powered by Open3d, an open source 3d python library. The point clouds from depth data is clustered and then borders were drawn for each clustered group.

for further information read the report or the flowchart inside docs folder.
The code was written and test on Asus zephyrus g14 with Intel realsense d455. the breif specs are as followed
- Ryzen 9 5900hs
- RTX 3060
- 16 gb of ram
- 2560 x 1440 display

# To run the code
1. install dependencies with `pip install -r requirements.txt`
2. make sure pytorch with gpu enabled and realsense SDK are installed
### With a recorded rgbd file 
1. head into recorded.py inside scripts folder and change the path on line 29th to your desire path
2. with the terminal on Navisense directory run `py ./scripts/recorded.py` 
### with the camera connected
- with the terminal on Navisense directory run `py ./scripts/realtime.py` for regular OpenCv windows
- with the terminal on Navisense directory run `py ./scripts/with_gui.py` for the graphic user interface windows version.
**the window might not fit your screen**

# Please note
1. make sure your gpu is Nvidia, has CUDA cores and sufficient vram (the written machine has 6gb)
2. if your interpreter is not `py` replace `py` with your python interpreter. for example `python3`
3. make sure your terminal directory is `.../Navisense`
4. if you have a problem with connecting camera, try using realsense SDK tools to see if you camera show up or has the right software
5. if you wish to change the camera to another Intel realsense model. change the setting inside rs_config.json in scripts folder

# folder description
- **Navisense** contains the functions necessary to run the code
- **PIDNet** contains the partial PIDNet model leaving out unnecessary files and scripts
- **docs** contains flowcharts of the system
- **scripts** contains scrips used to run the system





