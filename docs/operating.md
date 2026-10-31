<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Running the server under a restart policy

The server ends itself with exit code 70 when its engine dies, and leaves the restart to whatever runs it ([`running.md`](running.md#when-the-engine-dies)).
This page has one complete file for each of the three usual supervisors.
Each one restarts on the exit, and each one waits long enough for a first start, which includes the checkpoint download (the 31B's first start took 7 min 54 s with the download and 4 min 3 s with the weights on disk, [`README.md`](../README.md#gpu-server); the image's own health check allows 45 minutes).

What each file was checked against:

| File | State |
| --- | --- |
| Docker Compose | the repository's own `compose.yaml`, shown here; it brought the container up on a card on 2026-10-01 (`runs/2026-10-01_docker-first-gpu-start/`) |
| systemd | written for this page and **not run**; the options are systemd's documented ones |
| Kubernetes | written for this page and **not run**; the probe is the server's `/health` as measured on the stand-in and the card |

None of the three adds authentication: the server has none on any route, so keep the port on the loopback interface or behind your own proxy (`SECURITY.md`).

## Docker Compose

`compose.yaml` in the repository root (its comments shortened here):

```yaml
name: decisio

services:
  decisio:
    build:
      context: .
      args:
        APPLY_PATCHES: "0"            # "1" applies the optional suffix-staging series (patches/README.md)
    image: decisio:local
    environment:
      DECISIO_BASE: gemma-4-31b       # the default (measured on 96 GB, runs on 80 GB); qwen3.6-35b-a3b, gemma-4-12b or gemma-4-12b-fp8 for the others
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - decisio-data:/data
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped           # also after a dead engine: the server exits with code 70

volumes:
  decisio-data:
```

`docker compose up --build` after `uv build` builds the image from the `Dockerfile`; to run a published image, replace `build:` with `image: ghcr.io/aminry/decisio@sha256:<the digest in the release notes>`.
Docker restarts a container when its process exits, not when its `HEALTHCHECK` fails, so it is the exit with code 70 that brings the server back; the image's check (`/health`, every 30 seconds, a 45-minute start period) shows the container unhealthy in the meantime.
The named volume `decisio-data` keeps the checkpoint, so a restart does not download it again.

## systemd

For the GPU server install on a Linux host (`/opt/decisio` holding the checkout, set up once with `uv sync --extra serve --frozen` as in the [README](../README.md#gpu-server), and a user `decisio`):

```ini
# /etc/systemd/system/decisio.service
[Unit]
Description=decisio server
After=network-online.target
Wants=network-online.target

[Service]
User=decisio
WorkingDirectory=/opt/decisio
Environment=HF_HOME=/var/lib/decisio/hf
ExecStart=/usr/local/bin/uv run --no-sync python -m decisio.serve.vllm_engine --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
# a first start downloads the checkpoint and loads the engine: do not let systemd give up on it
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
```

`Restart=on-failure` restarts on any non-zero exit, which includes code 70; `systemctl stop` is a clean stop and is not restarted.
A repeated failure at start (a port in use, a card too small: [`troubleshooting.md`](troubleshooting.md)) is retried every 5 seconds until systemd's start limit stops it; `journalctl -u decisio` has the server's output.
`--no-sync` makes `uv run` use the environment the `uv sync` built, so that the server runs the versions the lock file pins.

## Kubernetes

A Deployment with one GPU, the checkpoint on a persistent volume, and the server's `/health` as the startup and liveness probe:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: decisio
spec:
  replicas: 1
  selector:
    matchLabels: {app: decisio}
  template:
    metadata:
      labels: {app: decisio}
    spec:
      restartPolicy: Always            # the default: the exit with code 70 starts a new container
      containers:
        - name: decisio
          image: ghcr.io/aminry/decisio@sha256:<the digest in the release notes>
          env:
            - {name: DECISIO_BASE, value: gemma-4-31b}
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - {name: data, mountPath: /data}
            - {name: shm, mountPath: /dev/shm}
          startupProbe:                # the first start downloads the checkpoint: 45 minutes, as the image's HEALTHCHECK
            httpGet: {path: /health, port: 8000}
            periodSeconds: 15
            failureThreshold: 180
          livenessProbe:               # /health answers 503 once the engine has died, before the process exits
            httpGet: {path: /health, port: 8000}
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: decisio-data}
        - name: shm
          emptyDir: {medium: Memory, sizeLimit: 8Gi}
```

Both probes read the same route: `/health` answers 200 once the engine is loaded and 503 with `{"ok": false, "engine": "dead", ...}` from the moment the engine is declared dead.
A readiness probe on `/health` is not needed for one replica; with several, remember that each replica is a separate server with its own registered tasks ([`tasks.md`](tasks.md)): a task taught on one is not on the others. `--tasks-dir` makes one server's tasks survive its restart (put the directory on a volume that outlives the container, and start with `--require-tasks N` so that a volume that failed to mount stops the server instead of leaving it untaught); it does not share tasks between replicas, and one directory belongs to one server. For replicas, start every one with `--task-store s3://bucket/prefix` (or another S3-compatible service, `--task-store-endpoint`): a task taught on one reaches the others within `--task-store-poll-seconds` (5), as the exact fitted record, and a second registration of the same task is refused with a 409 rather than overwriting it. Give the replicas the same stack if you certify (a certificate is suspended on a replica whose engine, card or settings differ), mark all but one `--task-store-readonly` and route the task-writing paths to it if you want fitting kept off the others, and sum the per-replica `decisio_task_*` metrics in Prometheus; a store outage stops teaching (503) and not serving ([`tasks.md`](tasks.md#replicas-a-shared-store---task-store)). A task with a fast table (`"fast_table": true`, [`tasks.md`](tasks.md#a-fast-table-in-front-of-decisio-opt-in)) needs `pip install 'decisio[table]'` on every replica and CPU headroom on each: the table runs on the host CPU, its timing against the model is made on the registering replica's stack, and a replica on another CPU, card or engine serves that task through the model until it has timed the table on its own requests (`status.fast_table` on each replica says which it is doing and why). A suspension of the table written by one replica reaches the others through the shared store within one poll; with `--tasks-dir` or a read-only replica it stays on the replica that saw the alarm.
Do not put a request on the probe path: `/health` costs nothing, and a question costs an engine call.
