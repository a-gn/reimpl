FROM ubuntu:24.04 AS main

    ENV DEBIAN_FRONTEND=noninteractive

    # basic tools + Python's runtime deps
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates \
            git \
            gnupg2 \
            htop \
            openssh-client \
            python3 \
            python3-venv \
            tmux

    RUN python3.12 -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    ENV VIRTUAL_ENV="/opt/venv"

    RUN python -m pip install -U pip && \
        python -m pip install -U poetry setuptools wheel

    RUN mkdir /workspace
    WORKDIR /workspace
