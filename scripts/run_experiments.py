"""GRU / GCN / 관계형 GCN 을 시드 3개로 학습해 그래프 효과를 확인한다.

결과: results/experiments.json (모델별 WAPE·MAE 평균·표준편차), results/models/*.pt, results/preds/*.npy
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieum_ai import forecast as F
from ieum_ai.graph import build_graph

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "models").mkdir(parents=True, exist_ok=True); (RES / "preds").mkdir(exist_ok=True)
log = lambda *a: print(*a, flush=True)

df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
p = F.prepare(df)

r = pd.read_csv("/home/claude/yongin/용인시_버스노선_경유정류소.csv", encoding="utf-8-sig", dtype={"노선ID": str, "정류소ID": str})
s = pd.read_csv("/home/claude/yongin/용인시_버스정류소_현황.csv", encoding="utf-8-sig", dtype={"정류소ID": str})
rs = r.rename(columns={"노선ID": "route_id", "진행구분코드": "direction", "정류소순서": "seq", "정류소ID": "stop_id"})
stops = pd.DataFrame({"stop_id": p.stop_ids}).merge(
    s.rename(columns={"정류소ID": "stop_id", "WGS84위도": "lat", "WGS84경도": "lon"})[["stop_id", "lat", "lon"]], how="left")
ids, ei, et = build_graph(stops, rs[["route_id", "direction", "seq", "stop_id"]])
N = len(ids)
A_all = F.normalized_adj(ei, N)
A_route = F.normalized_adj(ei[:, et == 0], N)
A_near = F.normalized_adj(ei[:, et == 1], N)

test = F.origins(p, "test")
configs = {"gru": (False, None), "gcn": (True, A_all), "rgcn": (True, [A_route, A_near])}
results = {k: [] for k in configs}
for seed in (0, 1, 2):
    for name, (g, adj) in configs.items():
        t = time.time()
        model = F.train_nn(p, use_graph=g, adj=adj, seed=seed, train_step=2, log=lambda *a: None)
        pred = F.nn_predict(model, p, test)
        sc = F.score(p, test, pred)
        results[name].append({k: float(v) for k, v in sc.items()})
        torch.save(model.state_dict(), RES / "models" / f"{name}_s{seed}.pt")
        np.save(RES / "preds" / f"{name}_s{seed}.npy", pred)
        log(f"{name} seed{seed} WAPE={sc['WAPE']:.4f} MAE={sc['MAE']:.2f} ({time.time()-t:.0f}s)")

summary = {k: {m: {"mean": float(np.mean([r[m] for r in v])), "std": float(np.std([r[m] for r in v]))}
               for m in ("WAPE", "MAE")} for k, v in results.items()}
json.dump({"runs": results, "summary": summary}, open(RES / "experiments.json", "w"), indent=2)
log("DONE", json.dumps(summary))
