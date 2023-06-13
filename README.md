# NaviSense
Navigation assistance for visually impaired person project using the combination of depth and color data from the RGBD camera to navigate 

# About the Project
the system use two different method to detect obstacle
1. floor level obstacle detection is powered by PIDNet, an image segmentation machine learning model. The model is then trained with additional 300+ images of sidewalks around Chulalongkorn University
2. upper floor level obstacle detection is powered by Open3d, an open source 3d python library. The point clouds from depth data is clustered and then borders were drawn
for further information read the report or the flowchart inside docs folder

# To run the code
1. install dependencies with `pip install -r requirements.txt`
2. make sure pythorch with gpu enable and realsense SDK are installed
### to read a recorded rgbd file 
1. head into recorded.py inside scripts folder and change the path on line 29th to your desire path
2. with the terminal on Navisense directory run `py ./scripts/recorded.py



