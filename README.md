# The Ariadne's Thread team repo

## Description

We built a simulation enviroment using ROS2 and Gazebo to test the performances of various planning algorithms. The test
scenario consists in collaborative planning between ground and aerial robots for planetary exploration. The ground robot
uses the aerial map generated by the flying robot to efficiently plan its path. The flying robot is responsible for
exploring the environment and generating a map of the obstacles the rover could encounter on its path.

## Requirements

To run the code you need to have ROS2 Humble installed on your machine. You can find the installation
instructions [here](https://docs.ros.org/en/humble/Installation.html).

## Running the code

To try out our code use the following command:

```bash
mkdir -p ariadne_ws/src
cd ariadne_ws/src
git clone git@github.com:viciopoli01/The-Ariadne-s-Thread.git
```

Then build the workspace with:

```bash
cd ..
colcon build
```

To run the code use the following command:

```bash
source install/setup.bash
ros2 launch ariadne start_ariadne.launch.py
```

## Docker usage

```bash
docker build -t ariadne:amd64 .
docker run --net=host --rm --privileged --name ARIADNE -it ariadne:amd64
```

## Authors

...