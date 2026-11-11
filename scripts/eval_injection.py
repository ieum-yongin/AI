"""가짜 급변을 심어서 알림 성능을 평가한다.

정답 라벨이 없으니, 테스트 기간(2026-06~08)의 정류소 일부에 인위적인 변화를 심는다.
    - 급감: 10일 동안 이용량 × 0.4
    - 급증: 10일 동안 이용량 × 1.8
그다음 기준선(예측값)마다 같은 알림 규칙을 적용해서
    재현율  : 심은 급변을 잡아낸 비율
    오탐률  : 아무것도 안 심은 정류소에 알림이 뜬 비율 (실제 사건도 섞여 있어서 '상한'으로 봄)
    탐지 지연: 변화 시작부터 첫 알림까지 걸린 날
을 비교한다. 알림 규칙: 실제/예측 비율의 7일 이동 중앙값 < 0.6 또는 > 1.6
"""
import copy, json, sys
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

df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
p0 = F.prepare(df)
N = len(p0.stop_ids)
test = F.origins(p0, "test")          # 1일 앞 예측의 대상 날짜 = test origins
D = len(test)

# 그래프 (학습 때와 같게)
r = pd.read_csv("/home/claude/yongin/용인시_버스노선_경유정류소.csv", encoding="utf-8-sig", dtype={"노선ID": str, "정류소ID": str})
s = pd.read_csv("/home/claude/yongin/용인시_버스정류소_현황.csv", encoding="utf-8-sig", dtype={"정류소ID": str})
rs = r.rename(columns={"노선ID": "route_id", "진행구분코드": "direction", "정류소순서": "seq", "정류소ID": "stop_id"})
stops = pd.DataFrame({"stop_id": p0.stop_ids}).merge(
    s.rename(columns={"정류소ID": "stop_id", "WGS84위도": "lat", "WGS84경도": "lon"})[["stop_id", "lat", "lon"]], how="left")
_, ei, et = build_graph(stops, rs[["route_id", "direction", "seq", "stop_id"]])
ADJ = {"gru": (False, None), "gcn": (True, F.normalized_adj(ei, N)),
       "rgcn": (True, [F.normalized_adj(ei[:, et == 0], N), F.normalized_adj(ei[:, et == 1], N)])}


def load(name, seed):
    g, adj = ADJ[name]
    m = F.Forecaster(N, use_graph=g, adj=adj)
    m.load_state_dict(torch.load(RES / "models" / f"{name}_s{seed}.pt"))
    return m.eval()


LEAD = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 기준선을 며칠 전 데이터로 만들지 (1~7)
TAG = f"_lead{LEAD}"


def one_day_ahead(kind, p):
    """날짜 d의 기준선 = d-LEAD 시점까지의 데이터로 만든 예측. LEAD가 크면 급변에 덜 끌려간다."""
    h = LEAD - 1
    org = test - h
    if kind == "seasonal":
        return np.expm1(F.seasonal_naive(p, org)[:, h])
    name, seed = kind.rsplit("_s", 1)
    return np.expm1(F.nn_predict(load(name, int(seed)), p, org)[:, h])


def alerts(p, base):
    act = p.raw[test]
    ratio = np.where(p.obs[test] & (base > 0), act / np.maximum(base, 1e-6), np.nan)
    r7 = pd.DataFrame(ratio).rolling(7, min_periods=5).median().to_numpy()
    return (r7 < 0.6) | (r7 > 1.6)                       # (D, N)


# 대상 정류소: 테스트 기간 평균 이용 30명 이상, 95% 이상 관측
obs_rate = p0.obs[test].mean(0)
mean_use = np.nanmean(p0.raw[test], 0)
eligible = np.where((obs_rate >= 0.95) & (mean_use >= 30))[0]

kinds = ["seasonal"] + [f"{m}_s{sd}" for m in ("gru", "gcn", "rgcn") for sd in (0, 1, 2)]
rows = []
for draw in range(3):
    rng = np.random.default_rng(100 + draw)
    pick = rng.choice(eligible, 150, replace=False)
    kind_of = {j: ("DECREASE" if i < 75 else "INCREASE") for i, j in enumerate(pick)}
    start = {j: int(rng.integers(7, D - 17)) for j in pick}
    p = copy.deepcopy(p0)
    for j in pick:
        f = 0.4 if kind_of[j] == "DECREASE" else 1.8
        days = test[start[j]:start[j] + 10]
        p.raw[days, j] = p.raw[days, j] * f
        p.y[days, j] = np.log1p(np.nan_to_num(p.raw[days, j]))
    clean = np.setdiff1d(eligible, pick)
    for kind in kinds:
        fl = alerts(p, one_day_ahead(kind, p))
        hit, delay = [], []
        for j in pick:
            w = fl[start[j]:start[j] + 13, j]
            hit.append(w.any())
            if w.any():
                delay.append(int(np.argmax(w)))
        dec = [h for h, j in zip(hit, pick) if kind_of[j] == "DECREASE"]
        inc = [h for h, j in zip(hit, pick) if kind_of[j] == "INCREASE"]
        rows.append({"draw": draw, "kind": kind.split("_s")[0], "seed": kind.split("_s")[1] if "_s" in kind else "-",
                     "recall": np.mean(hit), "recall_drop": np.mean(dec), "recall_rise": np.mean(inc),
                     "false_alarm": fl[:, clean].any(0).mean(), "delay_days": np.mean(delay) if delay else np.nan})
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(RES / f"injection_eval{TAG}.csv", index=False)
summ = out.groupby("kind")[["recall", "recall_drop", "recall_rise", "false_alarm", "delay_days"]].agg(["mean", "std"]).round(3)
print(summ.to_string())
summ.to_csv(RES / f"injection_summary{TAG}.csv")
