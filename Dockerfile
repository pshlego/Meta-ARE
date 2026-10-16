# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


# Stage 1 - Frontend build
FROM node:23 AS frontend-builder
WORKDIR /app/are/simulation/gui/client
COPY are/simulation/gui/client/package.json are/simulation/gui/client/package-lock.json .
# Clear npm cache and remove lock file to fix ARM64 rollup issue
RUN npm cache clean --force && rm -f package-lock.json
RUN --mount=type=cache,target=/root/.npm NPM_CONFIG_CACHE=/root/.npm npm install
COPY are/simulation/gui/client .
RUN npm run build

# Stage 2 - Python build
FROM python:3.12.6-slim AS python-builder
# Install uv
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install uv
WORKDIR /app
COPY pyproject.toml requirements.txt requirements-gui.txt requirements-dev.txt README.md LICENSE ./
COPY build_hooks ./build_hooks
COPY are/simulation /app/are/simulation
RUN rm -rf /app/are/simulation/gui/client
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -e ".[gui]"

# Stage 3 - Final stage
FROM python:3.12.6-slim
ARG SERVER_VERSION=unknown
WORKDIR /app

# Copy Python application and dependencies from python-builder stage
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY --from=python-builder /app /app

# Copy frontend build from frontend-builder stage
COPY --from=frontend-builder /app/are/simulation/gui/client/build /app/are/simulation/gui/client/build

EXPOSE 8080

# Env
ENV PYTHONUNBUFFERED=1
ENV ARE_SIMULATION_SSL_CERT_PATH=
ENV ARE_SIMULATION_SSL_KEY_PATH=
ENV ARE_SIMULATION_SERVER_HOSTNAME=
ENV ARE_SIMULATION_SERVER_PORT=8080
ENV ARE_SIMULATION_SERVER_VERSION=$SERVER_VERSION

# Run
CMD ["are-gui"]
