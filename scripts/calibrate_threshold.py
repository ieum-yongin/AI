"""알림 임계값 정하기: 목표 오탐률(기본 5%)에 맞는 |log 비율| 값을 구한다.

학습에 안 쓴 테스트 기간(2026-06~08, 91일)에서 관계형 GCN 3개(시드 0·1·2)의 1일 앞 예측으로
정류소별 최대 급변 점수를 구하고, 상위 5% 지점을 임계값으로 잡는다 (세 시드 평균).
→ "평범한 정류소 100곳 중 5곳 정도만 91일 동안 한 번이라도 알림이 뜬다"는 뜻.
결과: models/threshold.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieum_ai import forecast as F
from ieum_ai import pipeline as P

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/claude/yongin")
FA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
p = F.prepare(df)
rs, loc = P.load_network(DATA / "용인시_버스노선_경유정류소.csv", DATA / "용인시_버스정류소_현황.csv")
adj = P.rgcn_adj(p.stop_ids, rs, loc)
test = F.origins(p, "test")
elig = np.where((p.obs[test].mean(0) >= 0.95) & (np.nanmean(p.raw[test], 0) >= 30))[0]

thr = []
for seed in (0, 1, 2):
    m = F.Forecaster(len(p.stop_ids), use_graph=True, adj=adj)
    m.load_state_dict(torch.load(ROOT / f"results/models/rgcn_s{seed}.pt"))
    m.eval()
    s, _ = P.alert_score(p, test, P.baseline_1day(m, p, test))
    mx = np.nan_to_num(np.abs(s), nan=0.0)[:, elig].max(0)
    thr.append(float(np.quantile(mx, 1 - FA)))
    print(f"seed{seed}: threshold={thr[-1]:.3f} (비율 {np.exp(-thr[-1]):.2f}배 이하 / {np.exp(thr[-1]):.2f}배 이상)")

out = {"target_false_alarm": FA, "window_days": len(test), "abs_log_ratio": round(float(np.mean(thr)), 4),
       "decrease_below_ratio": round(float(np.exp(-np.mean(thr))), 3),
       "increase_above_ratio": round(float(np.exp(np.mean(thr))), 3),
       "min_mean_daily_usage": 30, "calibrated_on": f"{p.dates[test[0]].date()}~{p.dates[test[-1]].date()}",
       "per_seed": thr}
(ROOT / "models").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "models/threshold.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
