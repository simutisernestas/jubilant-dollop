FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install numpy>=1.21.5 transforms3d>=0.4.1 matplotlib>=3.5.1 scipy>=1.8.0

# Configure Matplotlib to use Agg backend
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" >> /root/.config/matplotlib/matplotlibrc

# Set working directory
WORKDIR /app

# Copy project files to working directory
COPY . /app

# Set the entrypoint for the container
ENTRYPOINT [ "bash" ]
