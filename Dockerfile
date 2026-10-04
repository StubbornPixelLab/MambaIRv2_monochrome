FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install packaging ninja numpy


WORKDIR /home/yinqiang/jixiang

CMD ["/bin/bash"]
