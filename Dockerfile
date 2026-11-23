# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# The decisio server on vLLM 0.32.0: the vLLM release image (pinned by digest) plus the decisio wheel.
#   uv build --wheel && docker build --platform linux/amd64 -t decisio:local .
# The wheel comes from the build context (dist/). The checkpoint is never part of the image: the first start fetches it
# into the volume mounted at /data. The image is stock vLLM: the optional latency patch series (patches/README.md) is
# for vLLM 0.31.0, and --build-arg APPLY_PATCHES=1 fails the build.
# The base is the vLLM tag with its manifest-list digest; Dependabot refreshes the digest (a tag bump is a measured change).
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

ARG APPLY_PATCHES=0

LABEL org.opencontainers.image.title="decisio" \
      org.opencontainers.image.description="A prefill-only decision server on vLLM: typed questions about a state, a probability distribution per question" \
      org.opencontainers.image.source="https://github.com/aminry/decisio" \
      org.opencontainers.image.licenses="Apache-2.0"

# The wheel and its new dependencies (scipy, obstore for the shared task store, --task-store, and tzdata for time-v1's zone rules). Everything already in the image is held at its version: they are installed
# with the image's packages as constraints, the wheel with --no-deps (a resolver would re-read vLLM's own metadata and
# could replace torch), and check_requirements.py then confirms that decisio[serve] is satisfied by what the image ships.
COPY dist/decisio-*.whl /tmp/wheel/
COPY docker/pin_installed.py docker/check_requirements.py docker/constraints.txt /tmp/docker/
RUN python3 /tmp/docker/pin_installed.py > /tmp/pins.txt \
 && uv pip install --system --break-system-packages --no-cache -c /tmp/pins.txt -c /tmp/docker/constraints.txt scipy obstore tzdata \
 && uv pip install --system --break-system-packages --no-cache --no-deps /tmp/wheel/decisio-*.whl \
 && python3 /tmp/docker/check_requirements.py \
 && python3 -c "import decisio, decisio.vllm_plugin, scipy, obstore, tzdata" \
 && rm -rf /tmp/wheel /tmp/pins.txt /tmp/docker

# The optional patch series is written for vLLM 0.31.0 and does not apply to 0.32.0 (patches/README.md): a build that
# asks for it fails here rather than shipping an image without it.
RUN if [ "$APPLY_PATCHES" = "1" ]; then \
        echo "APPLY_PATCHES=1: the suffix-staging series is for vLLM 0.31.0 and does not apply to this image's vLLM 0.32.0 (patches/README.md)" >&2; \
        exit 1; \
    fi

# A non-root user whose home, caches and Hugging Face cache live in the volume.
RUN useradd --system --uid 10001 --user-group --home-dir /data --no-create-home --shell /usr/sbin/nologin decisio \
 && mkdir -p /data/hf /data/cache \
 && chown -R decisio:decisio /data
COPY docker/entrypoint.sh /usr/local/bin/decisio-entrypoint
RUN chmod 0755 /usr/local/bin/decisio-entrypoint

ENV HOME=/data \
    HF_HOME=/data/hf \
    XDG_CACHE_HOME=/data/cache \
    VLLM_CACHE_ROOT=/data/cache/vllm \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    DECISIO_PORT=8000

VOLUME /data
EXPOSE 8000
USER decisio
WORKDIR /data

# /health answers once the engine is loaded, which on the first start follows the checkpoint download (about 36 GB).
# It answers 503 once the engine has died; the server then exits with code 70, and the exit, not this check, is what a
# restart policy acts on (docs/running.md, "When the engine dies").
HEALTHCHECK --interval=30s --timeout=10s --start-period=45m --retries=3 \
    CMD ["python3", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['DECISIO_PORT'], timeout=5)"]

ENTRYPOINT ["/usr/local/bin/decisio-entrypoint"]
