FROM osrf/ros:noetic-desktop-full-focal


# Install some dependencies
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN pip3 install notebook catkin-tools

# copy the current directory contents into the container at /app
RUN mkdir -p /app/src/The-Ariadne-s-Thread
COPY . /app/src/The-Ariadne-s-Thread

RUN . /opt/ros/noetic/setup.sh && cd /app && catkin build ariadne

# start jupyter notebook as the entrypoint
CMD ["jupyter", "notebook", "/app/src/The-Ariadne-s-Thread/planner_notebook.ipynb", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]