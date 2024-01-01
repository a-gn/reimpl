FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        htop \
        openssh-client \
        python3.11 \
        python3.11-venv \
        tmux

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
