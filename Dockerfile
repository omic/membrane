FROM python:3.7.7-stretch
MAINTAINER Jun Seok Lee <lee.junseok39@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -qq -y gcc make \
	git \
apt-transport-https ca-certificates build-essential \
libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 libav-tools alsa-utils


# check our python environment
RUN python --version
RUN python -m pip --version

# set the working directory for containers
WORKDIR  /usr/membrane/src/

# Installing python dependencies
# COPY docker/requirements.txt .
COPY model/modelv2/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
# COPY docker/. /usr/membrane/src/
COPY model/modelv2/. /usr/membrane/src/
RUN ls -la /usr/membrane/src/*

# Running Python Application
CMD ["python3", "/usr/membrane/src/membrane.py"]
