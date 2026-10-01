#!/bin/bash
# Repeat of gate 3.0 steps 3 to 5 with the image pushed by the v0.1.0 release (by digest), plus conformance as an extra.
set -u
D=sha256:756a13db4e03ad1855bb8d0f71ec6f471e1a857194c1341fcf5bd07cb70a02dd
IMG=ghcr.io/aminry/decisio@$D
cd /root/decisio
R=runs/2026-10-01_docker-first-gpu-start/pushed_image_0.1.0
mkdir -p "$R"
date -u +%FT%TZ > "$R/began.txt"
# compose.yaml with image: set instead of build: (checklist 3.0, step 11); same project name, so the same volume
python3 - <<PY
s = open("compose.yaml").read()
a = s.index("    build:")
b = s.index("    image: decisio:local")
s = s[:a] + s[b:].replace("    image: decisio:local", "    image: $IMG", 1)
open("compose.pushed.yaml", "w").write(s)
PY
cp compose.pushed.yaml "$R/compose.pushed.yaml"
t0=$(date +%s); docker pull "$IMG" 2>&1 | tee "$R/pull.log" | tail -3; echo "pull: $(( $(date +%s) - t0 )) s" | tee "$R/time_pull.txt"
docker image inspect "$IMG" > "$R/image_inspect.json"
docker image inspect "$IMG" --format '{{index .RepoDigests 0}}' | tee "$R/repo_digest.txt"
docker run --rm --gpus all --entrypoint nvidia-smi "$IMG" -L 2>&1 | tee "$R/gpu_in_container.txt"
t0=$(date +%s); date -u +%FT%TZ > "$R/start_began.txt"
docker compose -f compose.pushed.yaml up -d 2>&1 | tee "$R/compose_up.log"
until curl -sf http://127.0.0.1:8000/health > "$R/health.json"; do sleep 5; done
echo "time to healthy (pushed image, checkpoint already in the volume): $(( $(date +%s) - t0 )) s" | tee "$R/time_to_healthy.txt"
docker compose -f compose.pushed.yaml ps --format json > "$R/compose_ps.json"
docker compose -f compose.pushed.yaml exec -T decisio id | tee "$R/container_user.txt"
docker compose -f compose.pushed.yaml logs --no-color > "$R/server.log" 2>&1
nvidia-smi > "$R/nvidia_smi.txt"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$R/gpu_memory_serving.txt"
curl -sS -o "$R/example_response.json" -w "%{http_code}\n" http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d @runs/2026-10-01_docker-first-gpu-start/example_request.json | tee "$R/example_status.txt"
# extra: the conformance gates against the pushed image, client in a second container of the same image
mkdir -p /root/conf_out2; chmod 777 /root/conf_out2
t1=$(date +%s)
docker run --rm --network host --user root --entrypoint bash \
  -v /root/decisio/conformance_items.json:/items.json:ro -v /root/conf_out2:/out -v decisio_decisio-data:/data:ro \
  -e HF_HOME=/data/hf -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/tmp/ts "$IMG" -c "
    uv pip install --system --break-system-packages --no-cache --target /tmp/ts typesafe-sdk==0.7.0 > /out/pip_typesafe.log 2>&1 &&
    python3 -m decisio.serve.systemone_conformance --url http://127.0.0.1:8000 --items /items.json \
      --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --block-size 1056 --n 200 --out /out/conformance.json" 2>&1 | tee "$R/conformance.log"
echo "conformance wall time: $(( $(date +%s) - t1 )) s" | tee "$R/time_conformance.txt"
cp /root/conf_out2/conformance.json "$R/"
docker compose -f compose.pushed.yaml down 2>&1 | tail -2
date -u +%FT%TZ > "$R/ended.txt"
echo done > "$R/repeat.done"
