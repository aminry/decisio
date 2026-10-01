#!/bin/bash
# A/B: the same example request, twice, against each image started from the same volume, one at a time.
cd /root/decisio
O=/root/ab_out; mkdir -p $O; rm -f $O/*
R=runs/2026-10-01_docker-first-gpu-start
run_one () {   # $1 label, $2 compose file
  docker compose -f "$2" up -d > /dev/null 2>&1
  until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
  for n in 1 2; do
    curl -sS -o $O/$1_$n.json http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d @$R/example_request.json
  done
  # and the first request again after one unrelated request, to see history effects
  curl -sS -o /dev/null http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{"state":"Hello","questions":{"q":{"type":"noul","instructions":"Is this a greeting?"}}}'
  curl -sS -o $O/$1_3.json http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d @$R/example_request.json
  docker compose -f "$2" down > /dev/null 2>&1
}
run_one local compose.yaml
run_one pushed compose.pushed.yaml
echo done > $O/ab.done
