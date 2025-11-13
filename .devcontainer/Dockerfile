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
            tmux \
            wget

    RUN wget -O /opt/lazygit_release.tar.gz https://github.com/jesseduffield/lazygit/releases/download/v0.56.0/lazygit_0.56.0_linux_x86_64.tar.gz && \
        echo "ced13c2ae074bbf6c201bc700ee2971e193b811c3b2ae0ed4d00d6225c6c9ab7  /opt/lazygit_release.tar.gz" | sha256sum -c - && \
        tar xf /opt/lazygit_release.tar.gz -C /usr/local/bin/ -- lazygit

    SHELL ["/usr/bin/fish", "-c"]

    # install uv
    RUN bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

    RUN mkdir /workspace
    WORKDIR /workspace
