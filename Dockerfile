FROM python:3.7.7-stretch
MAINTAINER Jun Seok Lee <lee.junseok39@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -qq -y gcc make \
	git \
apt-transport-https ca-certificates build-essential \
libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 libav-tools alsa-utils
# RUN apt-get install -y portaudio19-dev libopenblas-base libopenblas-dev pkg-config git-core cmake python-dev liblapack-dev libatlas-base-dev libblitz0-dev libboost-all-dev libhdf5-serial-dev libqt4-dev libsvm-dev libvlfeat-dev  python-nose python-setuptools python-imaging build-essential libmatio-dev python-sphinx python-matplotlib python-scipy
# additional dependencies
# RUN apt-get install -y \
#         libasound2 \
#         libasound-dev \
#         libssl-dev

# RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 libav-tools
# RUN apt-get install portaudio19-dev python-pyaudio python3-pyaudio

# RUN apt-get install -qq -y libportaudio2

# RUN pip install pyaudio

# check our python environment
RUN python --version
RUN python -m pip --version

# set the working directory for containers
WORKDIR  /usr/membrane/src/

# Installing python dependencies
# COPY docker/requirements.txt .
COPY modelv2/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
# COPY docker/. /usr/membrane/src/
COPY modelv2/. /usr/membrane/src/
RUN ls -la /usr/membrane/src/*

# Running Python Application
CMD ["python3", "/usr/membrane/src/membrane.py"]
