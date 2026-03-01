FROM ghcr.io/prefix-dev/pixi:latest AS build

WORKDIR /app

# Copy manifest files (linux-64 + linux-aarch64 lock file)
COPY pyproject.toml pixi.lock ./
COPY src/ ./src/

# Install dependencies into the pixi environment
RUN pixi install --locked

# Use the pixi shell-hook to create a self-contained activation script
RUN pixi shell-hook -s bash > /app/activate.sh

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install only the shared libraries that the Python ecosystem needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the populated pixi environment and the activation script from build
COPY --from=build /app/.pixi /app/.pixi
COPY --from=build /app/activate.sh /app/activate.sh
COPY --from=build /app/src /app/src
COPY --from=build /app/pyproject.toml /app/pyproject.toml

EXPOSE 8000

# Source the activation script so the pixi env is on PATH, then start uvicorn
CMD ["/bin/bash", "-c", "source /app/activate.sh && uvicorn simple_rag.api:app --host 0.0.0.0 --port 8000"]
