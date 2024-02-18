FROM ubuntu:22.04 AS python_builder

    RUN bash -c "echo 'deb-src http://archive.ubuntu.com/ubuntu/ jammy main' | tee -a /etc/apt/sources.list"

    ENV DEBIAN_FRONTEND=noninteractive

    RUN apt-get update \
        && \
        apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            git \
            pkg-config \
        && \
        apt-get build-dep -y python3 \
        && \
        apt-get install -y --no-install-recommends \
            build-essential gdb lcov pkg-config \
            libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
            libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
            lzma lzma-dev tk-dev uuid-dev zlib1g-dev

    RUN curl https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tar.xz -o /opt/python_src.tar.xz
    RUN mkdir /opt/python_src
    WORKDIR /opt/python_src
    RUN tar -xf /opt/python_src.tar.xz --strip-components=1
    RUN ./configure --enable-optimizations
    RUN make -j$(nproc)
    RUN make install


FROM ubuntu:22.04 AS main

    ENV DEBIAN_FRONTEND=noninteractive

    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates \
            git \
            gnupg2 \
            htop \
            openssh-client \
            tmux

    # install Python
    COPY --from=python_builder /usr/local/ /usr/local/
    # install Python's runtime dependencies
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            $(apt-cache depends libpython3.11-stdlib | grep Depends | sed "s/.*ends:\ //" | tr '\n' ' ')

    RUN python3.11 -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    ENV VIRTUAL_ENV="/opt/venv"

    RUN python -m pip install -U pip && \
        python -m pip install -U poetry setuptools wheel

    RUN mkdir /opt/poetry-app
    WORKDIR /opt/poetry-app
    COPY pyproject.toml poetry.lock /opt/poetry-app/
    RUN poetry install --no-interaction --no-root
    WORKDIR /

    COPY ./requirements-dev.txt /opt/requirements-dev.txt
    RUN pip install -r /opt/requirements-dev.txt
