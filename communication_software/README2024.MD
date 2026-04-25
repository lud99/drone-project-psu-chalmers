**Below might be not update and is from year 2024**
The ROS.py file depends on certain dependencies found in ATOS so make sure that you have ATOS installed. When ATOS is installed, do the follwing:

# First time setup
`cd atos_ws/src`

`ln -s /home/$USERNAME$/az-drone-safety-platform/communication_software/communication_software`
(The path should be the path to the folder containing this readme: `ln -s PATHTOTHISFOLDER`)

`cd ..`
`cd ..`

`cd az-drone-safety-platform/communication_software`

# Building
For the first time running a script, you have to perform these steps. __When changes are made in the code__, you have to perform these steps again.

`colcon build`

`source install/local_setup.bash`

# Running scripts
`ros2 run communication_software ENTRYPOINT`

for example:
`ros2 run communication_software run_main`


# Adding entry points
If you want to add new entry points (scripts to start from terminal), you can do so by adding them in "entry_points" in "setup.py". They should be in the following format:
`PATH.FILENAME:METHODNAME` where PATH is the relativee path seen from setup.py
For example:
`run_main = communication_software.main:main`
