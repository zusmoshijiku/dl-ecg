# Use an official PyTorch image as the base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python libraries for ECG analysis and deep learning
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    wfdb \
    neurokit2 \
    jupyter \
    tensorboard \
    tqdm \
    plotly \
    opencv-python-headless \
    albumentations

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/notebooks /app/src

# Copy your local files into the container
COPY . /app

# Expose port for Jupyter notebook
EXPOSE 8888

# Set the default command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
