# https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile

# Multi-stage build: First build the application with uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
 --mount=type=bind,source=uv.lock,target=uv.lock \
 --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
 uv sync --frozen --no-install-project --no-dev

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
 uv sync --frozen --no-dev

# For AMD64: Replace with CPU-only PyTorch to avoid CUDA bloat
# For ARM64: PyPI already provides CPU-only, so this will be a no-op
RUN uv pip uninstall -y torch torchvision && \
 uv pip install torch torchvision \
 --index-url https://download.pytorch.org/whl/cpu \
 --extra-index-url https://pypi.org/simple

# Then, use a final image without uv
FROM python:3.13-slim-bookworm
# It is important to use the image that matches the builder, as the path to the
# Python executable must be the same.

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Pre-download ResNet-50 weights during build to avoid cold start
RUN python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT)"

# Expose port
EXPOSE 8000

CMD ["fastapi", "run", "app.py", "--port", "8000", "--proxy-headers"]