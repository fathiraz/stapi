# Build stage
FROM python:3.10-slim-bookworm as builder

# Set environment variables
ENV PORT=8080 \
    MODEL=multi-qa-MiniLM-L6-cos-v1 \
    MAX_BATCH_SIZE=100 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get -y --no-install-recommends install python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy application files
COPY main.py .

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/

# Run entrypoint script
CMD [ "entrypoint.sh" ]

# Expose port
EXPOSE ${PORT}
