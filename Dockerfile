# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script to train the model during build
RUN python train.py

# Define environment variable
ENV ENVIRONMENT=production

# Run test.py when the container launches
CMD ["python", "test.py"]