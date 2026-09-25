# Multi-stage build for Runner SDK services (local-cluster deploy)
# Supports modes: aer, tool, memory-server
#
# Build context: client-libraries/
# Usage: docker buildx build -t runner:latest -f packages/python/packages/agent-engine-runner-shared/Dockerfile .

ARG BASE_TARGETPLATFORM=linux/amd64
ARG RUNNER_BASE_IMAGE=664315256653.dkr.ecr.us-east-1.amazonaws.com/base/amazonlinux2023-python:v3.11-202512082043-f9f77a3

FROM --platform=${BASE_TARGETPLATFORM} ghcr.io/astral-sh/uv@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 AS uv

# Build stage - Install dependencies
FROM --platform=${BASE_TARGETPLATFORM} ${RUNNER_BASE_IMAGE} AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /build

# Copy workspace-level manifests first (these change less often than source code).
COPY packages/python/pyproject.toml packages/python/uv.lock packages/python/uv.toml ./

# Copy dependency manifests for workspace packages.
COPY packages/python/packages/agent-engine-sdk-memory/pyproject.toml ./packages/agent-engine-sdk-memory/
COPY packages/python/packages/agent-engine-sdk/pyproject.toml ./packages/agent-engine-sdk/
COPY packages/python/packages/agent-engine-runner-shared/pyproject.toml packages/python/packages/agent-engine-runner-shared/README.md ./packages/agent-engine-runner-shared/
COPY packages/python/packages/agent-engine-sdk-langgraph/pyproject.toml packages/python/packages/agent-engine-sdk-langgraph/README.md ./packages/agent-engine-sdk-langgraph/
COPY packages/python/examples/insurance-agent/pyproject.toml packages/python/examples/insurance-agent/README.md ./examples/insurance-agent/

# Stub out source dirs so uv can resolve workspace packages during sync.
RUN mkdir -p packages/agent-engine-sdk-memory/agent_engine_sdk_memory \
    && touch packages/agent-engine-sdk-memory/agent_engine_sdk_memory/__init__.py \
    && mkdir -p packages/agent-engine-sdk/agent_engine_sdk \
    && touch packages/agent-engine-sdk/agent_engine_sdk/__init__.py \
    && mkdir -p packages/agent-engine-runner-shared/src/agent_engine_runner_shared \
    && touch packages/agent-engine-runner-shared/src/agent_engine_runner_shared/__init__.py \
    && mkdir -p packages/agent-engine-sdk-langgraph/src/agent_engine_sdk_langgraph \
    && touch packages/agent-engine-sdk-langgraph/src/agent_engine_sdk_langgraph/__init__.py \
    && mkdir -p examples/insurance-agent/src/insurance_agent \
    && touch examples/insurance-agent/src/insurance_agent/__init__.py

# Install dependencies using uv.
# This layer is cached when only source code changes.
WORKDIR /build/examples/insurance-agent
RUN uv sync --frozen --no-dev --no-install-workspace --compile-bytecode

# Copy actual source code last — changes more frequently than dependencies.
WORKDIR /build
COPY packages/python/packages/agent-engine-sdk-memory/agent_engine_sdk_memory ./packages/agent-engine-sdk-memory/agent_engine_sdk_memory/
COPY packages/python/packages/agent-engine-sdk/agent_engine_sdk ./packages/agent-engine-sdk/agent_engine_sdk/
COPY packages/python/packages/agent-engine-runner-shared/src ./packages/agent-engine-runner-shared/src/
COPY packages/python/packages/agent-engine-sdk-langgraph/src ./packages/agent-engine-sdk-langgraph/src/
COPY packages/python/examples/insurance-agent/src ./examples/insurance-agent/src/

# Install workspace packages now that source is present.
WORKDIR /build/examples/insurance-agent
RUN uv sync --frozen --no-dev --compile-bytecode

# Runtime stage
FROM --platform=${BASE_TARGETPLATFORM} ${RUNNER_BASE_IMAGE} AS runtime

RUN set -eux; \
    if command -v yum >/dev/null 2>&1; then \
      yum install -y curl && yum clean all; \
    elif command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*; \
    else \
      echo "Unsupported base image package manager" >&2; exit 1; \
    fi; \
    if ! getent group app-user >/dev/null 2>&1; then groupadd -g 1001 app-user; fi; \
    if ! id -u app-user >/dev/null 2>&1; then useradd -m -u 1001 -g app-user app-user; fi

WORKDIR /app

# Copy the entire venv from builder (includes all workspace packages).
# uv places the venv at the workspace root, not the package directory.
COPY --from=builder --chown=app-user:app-user /build/.venv /app/.venv

# Copy insurance agent source
COPY --from=builder --chown=app-user:app-user /build/examples/insurance-agent /app/insurance-agent

USER app-user

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT:-8000}/health || exit 1

ENV RUNNER_MODE=aer

WORKDIR /app/insurance-agent
CMD ["python", "-m", "insurance_agent.main"]
