FROM pytorch/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore


# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    locales \
    python3-opencv \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    byobu \
    htop \
    vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8


RUN pip install --upgrade pip
RUN pip install flake8 typing mypy pytest pytest-mock
RUN pip install ufmt==1.3.2 black==22.3.0 usort==1.0.2
RUN pip install pre-commit

ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspaces/gptmm
RUN /bin/bash -c python setup.py develop
