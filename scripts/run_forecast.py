"""비교 기준 3개 + GNN 학습·평가. 실행: python scripts/run_forecast.py"""
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
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
log = lambda *a: print(*a, flush=True)

df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
p = F.prepare(df)
log(f"정류소 {len(p.stop_ids)}곳, {len(p.dates)}일, 학습~{p.dates[p.split[0]-1].date()}, 검증~{p.dates[p.split[1]-1].date()}, 테스트~{p.dates[-1].date()}")

# 그래프 (노선 연결 + 300m)
r = pd.read_csv("/home/claude/yongin/용인시_버스노선_경유정류소.csv", encoding="utf-8-sig", dtype={"노선ID": str, "정류소ID": str})
s = pd.read_csv("/home/claude/yongin/용인시_버스정류소_현황.csv", encoding="utf-8-sig", dtype={"정류소ID": str})
rs = r.rename(columns={"노선ID": "route_id", "진행구분코드": "direction", "정류소순서": "seq", "정류소ID": "stop_id"})
stops = pd.DataFrame({"stop_id": p.stop_ids}).merge(
    s.rename(columns={"정류소ID": "stop_id", "WGS84위도": "lat", "WGS84경도": "lon"})[["stop_id", "lat", "lon"]], how="left")
ids, ei, et = build_graph(stops, rs[["route_id", "direction", "seq", "stop_id"]])
deg = np.bincount(ei[0], minlength=len(ids))
log(f"그래프: 엣지 {ei.shape[1]} (노선 {int((et==0).sum())}, 거리 {int((et==1).sum())}), 연결 없는 정류소 {int((deg==0).sum())}")
adj = F.normalized_adj(ei, len(ids))

test = F.origins(p, "test")
res = {}

t = time.time(); res["seasonal"] = F.score(p, test, F.seasonal_naive(p, test)); log("seasonal", res["seasonal"], f"{time.time()-t:.0f}s")

t = time.time(); m = F.lgbm_model(p); res["lgbm"] = F.score(p, test, F.lgbm_predict(m, p, test)); log("lgbm", res["lgbm"], f"{time.time()-t:.0f}s")

for name, g in (("gru", False), ("gnn", True)):
    t = time.time(); log(f"[{name}]")
    model = F.train_nn(p, use_graph=g, adj=adj, log=log)
    pred = F.nn_predict(model, p, test)
    res[name] = F.score(p, test, pred); log(name, res[name], f"{time.time()-t:.0f}s")
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    if g:
        np.save(OUT / "gnn_test_pred.npy", pred)

json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, open(OUT / "forecast_scores.json", "w"), indent=2)
log(json.dumps(res, default=float, indent=1))
