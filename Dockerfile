# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:10.1-devel-centos7
LABEL name="desmiles"
LABEL version="1.0"

RUN yum -y install git  

RUN git clone https://github.com/DEShawResearch/DESMILES /opt/DESMILES
# alternatively, if you already have a local repo,
# assuming you are starting one level above the local repo,
# you can use:
# COPY DESMILES /opt/DESMILES

# set up conda
ENV CONDA_DIR /opt/conda
RUN curl -s https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
-o /opt/Miniconda3-py37_5.10.3-Linux-x86_64.sh 
RUN /bin/bash /opt/Miniconda3-py37_5.10.3-Linux-x86_64.sh  -b -p /opt/conda 
ENV PATH=$CONDA_DIR/bin:$PATH

# setup conda env
# see above about where environment.yml will be found
RUN conda env create --file /opt/DESMILES/environment.yml
# make useable in build
RUN source activate desmiles

RUN echo "source activate desmiles" > ~/.bashrc
# for 0.4rc install/source/run as user

# DESMILES requires this environment variable.
# set to recommended mountpoint for desmiles. 
ENV DESMILES_DATA_DIR=/desmiles/data

# If your local data directory is PATH_TO_DATA_DIR,
# add the option -v PATH_TO_DATA_DIR:desmiles/data

# If you chose to mount the data elsewhere,
# set`-e DESMILES_DATA_DIR`  accordingly.

# If you know how to use docker, the above information will get you started. 
# Build this image with:
# docker build -t desmiles:1.0 https://github.com/DEShawResearch/DESMILES.git#1.0
# Assuming that this docker image builds correctly,
# then you can start this docker image using:
# docker run -p 8888:8888 -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v PATH_TO_DATA_DIR:/desmiles/data desmiles:0.5a 
