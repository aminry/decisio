#!/bin/bash
cd /root/decisio
O=/root/rep_out; mkdir -p $O; rm -f $O/*
R=runs/2026-10-01_docker-first-gpu-start
docker compose -f compose.pushed.yaml up -d > /dev/null 2>&1
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
python3 - <<'PY'
import json, urllib.request
req = json.load(open("runs/2026-10-01_docker-first-gpu-start/example_request.json"))
def post(body):
    r = urllib.request.Request("http://127.0.0.1:8000/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))["answers"]
out = {"full": [], "urgent_only": [], "category_only": [], "impact_only": []}
for i in range(30):
    a = post(req)
    out["full"].append([a["urgent"]["noul"], a["category"]["probabilities"]["access"], a["impact"]["score"]])
    for name, q in (("urgent_only", "urgent"), ("category_only", "category"), ("impact_only", "impact")):
        b = post({"state": req["state"], "questions": {q: req["questions"][q]}})[q]
        out[name].append(b["noul"] if q == "urgent" else (b["probabilities"]["access"] if q == "category" else b["score"]))
json.dump(out, open("/root/rep_out/rep30.json", "w"))
PY
docker compose -f compose.pushed.yaml down > /dev/null 2>&1
echo done > $O/rep.done
