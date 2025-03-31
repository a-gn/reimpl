FROM ubuntu:24.04 AS main

    ENV DEBIAN_FRONTEND=noninteractive

    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            fish \
            git \
            htop \
            openssh-client \
            tmux

    SHELL ["/usr/bin/fish", "-c"]

    # install uv
    RUN bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

    RUN mkdir /workspace
    WORKDIR /workspace
