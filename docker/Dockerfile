# syntax=docker/dockerfile:experimental
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
ENV TZ="UTC"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends nano tmux git wget unzip bzip2 sudo && \
    apt-get install -y ssh libsm6 libxext6 imagemagick awscli ffmpeg && \
    conda install -y -c conda-forge ffmpeg && \
    apt-get clean

COPY training-requirements.txt /workspace/
### Install application including those in Aquabyte PyPI
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
ENV PIP_EXTRA_INDEX_URL=http://aquabyte-repository.s3-website-us-west-1.amazonaws.com/pypi/
ENV PIP_TRUSTED_HOST=aquabyte-repository.s3-website-us-west-1.amazonaws.com
RUN --mount=type=ssh pip install --no-cache-dir -U pip setuptools wheel && pip install -r training-requirements.txt

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ARG USER_UID
ARG USERNAME
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && adduser $USERNAME sudo
    
COPY start_jupyter.sh /workspace/

SHELL ["/bin/bash", "-c"]