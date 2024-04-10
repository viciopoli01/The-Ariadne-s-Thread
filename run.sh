#!/bin/bash

echo "          _____  _____          _____  _   _ ______ "
echo "    /\   |  __ \|_   _|   /\   |  __ \| \ | |  ____|"
echo "   /  \  | |__) | | |    /  \  | |  | |  \| | |__   "
echo "  / /\ \ |  _  /  | |   / /\ \ | |  | | . \` |  __| "
echo " / ____ \| | \ \ _| |_ / ____ \| |__| | |\  | |____ "
echo "/_/    \_\_|  \_\_____/_/    \_\_____/|_| \_|______|"
echo "                                                    "

echo "Converting you notebook into a python script..."
jupyter nbconvert planner_notebook.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove_cell --to python
echo "Moving the python script to the ariadne/scripts/include directory..."
mv planner_notebook.py ariadne/scripts/include 

echo "Launching the Ariadne's Thread planning tool..."

COMMAND="roscore"

# Check if the command is already running
if pgrep -x "$COMMAND" >/dev/null; then
    echo "Command $COMMAND is already running."
else
    (roscore)
fi


cd /app && catkin build ariadne && . devel/setup.bash && roslaunch ariadne ariadne.launch & rqt
# Check the exit status
if [ $? -ne 0 ]; then
    RED='\033[0;31m'
    NC='\033[0m'
    echo "${RED}Make sure you have built the workspace."
    echo "If you have not, please run the following commands:"
    echo "cd YOUR_WORKSPACE"
    echo "catkin build ariadne"
    echo "Also, make sure the folder structure is correct."
    echo "The run.sh file should be in the YOUR_WORKSPACE/src/The-Ariadne-s-Thread directory.${NC}"
fi