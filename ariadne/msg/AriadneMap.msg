std_msgs/Header header

geometry_msgs/Point goal  # The goal to plan to, both the robots known their positions in a common frame

float64[] temp_goal

float64[] obstacles_radius_list # one radius for each obstacle
geometry_msgs/Point[] obstacles_coordinate_list # The obstacles to avoid