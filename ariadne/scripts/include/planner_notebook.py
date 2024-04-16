#!/usr/bin/env python
# coding: utf-8

# # Yuo can write your planner here:

# Do not modify the name of the function. You can add any number of helper functions.

# In[ ]:


def planner(start, goal, obstacles: list, radius: list) -> list:
    """ Plan a path from start to goal avoiding obstacles.

    Args:
        start (list): list of x, y coordinates of the start point
        goal (list): list of x, y coordinates of the goal point
        obstacles (list): list of obstacles, each obstacle is a tuple (x, y, 0)
        radius (list): list of radius of obstacles
    
    Returns:
        list: This must be a valid list of connected nodes that form
            a path from start to goal node
    """
    
    return []


# Then execute the following cell to run the tests.

# In[ ]:


get_ipython().system('(roscore&)')


# In[ ]:


get_ipython().system('bash run.sh')

