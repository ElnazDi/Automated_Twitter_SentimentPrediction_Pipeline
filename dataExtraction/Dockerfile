FROM python:3.8.2-slim

# Set up anaconda variables
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Update packages in linux
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Set the working directory to /user/src/models
WORKDIR /usr/src/models

# Copy file with dependencies
COPY . /usr/src/models

# Install python dpendencies
RUN pip install -r requirementsMlflow.txt

# Start the server CMD
ENTRYPOINT ["mlflow", "server"]