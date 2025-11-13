#!/bin/bash

echo "Configuring git user identity..."
git config --global user.name "Arno Gobbin"
git config --global user.email "32413451+a-gn@users.noreply.github.com"

echo "Installing project with uv..."
uv sync --dev

echo "Setting up nbstripout to clear the output of notebook cells before committing..."
uv run nbstripout --install

