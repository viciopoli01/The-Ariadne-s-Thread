FROM osrf/ros:noetic-desktop-full-focal


# Install some dependencies
RUN apt get-update
RUN apt-get install -y jupyer-core jupyter-notebook

# copy the current directory contents into the container at /app
COPY . /app

# start jupyter notebook as the entrypoint
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]