"""월간 분석 배치: 승하차 데이터 → 백엔드 테이블에 넣을 CSV.

사용:
    python scripts/run_batch.py --from 2026-06-01 --to 2026-08-31 --out out/2026-08
    (--model 생략 시 models/rgcn_v1.pt, --data 생략 시 data/stop_daily_usage.csv.gz)

출력 (정류소는 source_stop_id 기준. 백엔드가 bus_stop에서 stop_id로 바꿔 넣는다)
    analysis_run.json        분석 실행 1건 (status는 백엔드가 모두 넣은 뒤 COMPLETED로)
    stop_diagnosis.csv       모든 정류소의 이용 진단. 위험도 관련 값은 마을버스 노선 전까지 UNKNOWN/빈칸
    usage_alert.csv          이용 급변 알림 (alert_key = source_stop_id + change_start)
    alert_daily_detail.csv   알림별 일별 그래프 값 (alert_key로 연결)

알림 규칙
    기준선 = 관계형 GCN의 1일 앞 예측. 실제/기준선 비율의 7일 이동 중앙값이
    models/threshold.json 의 값(목표 오탐 5%)을 넘는 날이 3일 이상 이어지면 알림.
    7일 중앙값은 늦게 반응하므로 시작·종료일을 3일 앞당겨 기록한다.
"""
import argparse, json, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieum_ai import forecast as F
from ieum_ai import pipeline as P

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
LAG = 3          # 7일 이동 중앙값의 반응 지연 보정
MIN_DAYS = 3     # 연속으로 넘어야 하는 날 수
MAX_GAP = 2      # 이 이하로 끊기면 같은 알림으로 합침
PAD = 7          # 일별 그래프 앞뒤 여유

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", required=True)
ap.add_argument("--to", dest="end", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=str(ROOT / "models/rgcn_v1.pt"))
ap.add_argument("--data", default=str(ROOT / "data/stop_daily_usage.csv.gz"))
ap.add_argument("--network-dir", default="/home/claude/yongin")
ap.add_argument("--threshold", default=str(ROOT / "models/threshold.json"))
ap.add_argument("--stats-days", type=int, default=91, help="평균 승하차를 낼 최근 일수")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(a.data, dtype={"source_stop_id": str}, parse_dates=["service_date"])
rs, loc = P.load_network(Path(a.network_dir) / "용인시_버스노선_경유정류소.csv",
                         Path(a.network_dir) / "용인시_버스정류소_현황.csv")
model, bundle = P.load_bundle(a.model, rs, loc)
thr_cfg = json.load(open(a.threshold))
THR = thr_cfg["abs_log_ratio"]

# ------------------------------------------------------------ 1. 급변 알림
p = P.align(df, bundle["stop_ids"], bundle["node_mean"])
d0, d1 = pd.Timestamp(a.start), pd.Timestamp(a.end)
first = max(F.IN_LEN, p.dates.get_loc(d0) - 6 - LAG)            # 이동 중앙값 준비 기간 포함
days = np.arange(first, p.dates.get_loc(d1) + 1)
base = P.baseline_1day(model, p, days)
score, ratio = P.alert_score(p, days, base)
hist = slice(max(0, first - 28), first)
eligible = np.nanmean(p.raw[hist], 0) >= thr_cfg.get("min_mean_daily_usage", 30)
off = p.cal[:, 7] > 0


def episodes(flag):
    idx = np.where(flag)[0]
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= MAX_GAP + 1:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return [(g[0], g[-1]) for g in groups if g[-1] - g[0] + 1 >= MIN_DAYS]


alerts, detail = [], []
for j in np.where(eligible)[0]:
    sid = p.stop_ids[j]
    for direction, flag in (("DECREASE", score[:, j] < -THR), ("INCREASE", score[:, j] > THR)):
        for s, e in episodes(np.nan_to_num(flag, nan=False)):
            s, e = max(0, s - LAG), max(0, e - LAG)
            if p.dates[days[e]] < d0:                            # 보고 기간 전에 끝난 알림은 제외
                continue
            span = days[s:e + 1]
            b, o = base[s:e + 1, j], p.raw[span, j]
            ok = ~np.isnan(o)
            b_mean, o_mean = float(np.mean(b[ok])), float(np.mean(o[ok]))
            dr = ratio[s:e + 1, j]
            r = float(np.nanmedian(dr)) if np.isfinite(dr).any() else None   # 일별 비율의 중앙값 (요일 차이에 덜 흔들림)
            if r is None or (direction == "DECREASE") != (r < 1):           # 기간 보정 후 방향이 안 맞으면 버림
                continue
            n = e - s + 1
            holiday = off[span].mean() >= 0.5
            key = f"{sid}_{p.dates[span[0]].date()}"
            pct = abs(1 - r) * 100
            alerts.append({
                "alert_key": key, "source_stop_id": sid,
                "change_start": str(p.dates[span[0]].date()), "change_end": str(p.dates[span[-1]].date()),
                "change_direction": direction,
                "classification": "EXPECTED" if holiday else "NEEDS_CHECK",
                "expected_reason": "공휴일·명절 (용인 전체 이용 급감 기간)" if holiday else None,
                "usage_ratio": round(r, 3),
                "consecutive_days": n,
                "baseline_daily_usage": round(b_mean, 1), "observed_daily_usage": round(o_mean, 1),
                "analysis_summary": f"예측한 평소 이용량 대비 {pct:.0f}% {'감소' if direction == 'DECREASE' else '증가'}, {n}일 지속",
            })
            lo, hi = max(0, s - PAD), min(len(days) - 1, e + PAD)
            for k in range(lo, hi + 1):
                lr = np.log(ratio[k, j]) if ratio[k, j] > 0 else np.nan
                anom = s <= k <= e and not np.isnan(lr) and (lr < -THR if direction == "DECREASE" else lr > THR)
                obs_v = p.raw[days[k], j]
                detail.append({"alert_key": key, "service_date": str(p.dates[days[k]].date()),
                               "baseline_usage": round(float(base[k, j]), 1),
                               "observed_usage": None if np.isnan(obs_v) else int(obs_v),
                               "is_anomalous": bool(anom)})

alert_cols = ["alert_key", "source_stop_id", "change_start", "change_end", "change_direction", "classification",
              "expected_reason", "usage_ratio", "consecutive_days", "baseline_daily_usage",
              "observed_daily_usage", "analysis_summary"]
pd.DataFrame(alerts, columns=alert_cols).to_csv(out / "usage_alert.csv", index=False, encoding="utf-8")
pd.DataFrame(detail, columns=["alert_key", "service_date", "baseline_usage", "observed_usage", "is_anomalous"]) \
    .to_csv(out / "alert_daily_detail.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------ 2. 정류소 이용 진단 (모든 정류소)
win = df[(df.service_date > d1 - pd.Timedelta(days=a.stats_days)) & (df.service_date <= d1)]
g = win.groupby("source_stop_id").agg(B=("boarding_count", "mean"), A=("alighting_count", "mean"),
                                      Tb=("transfer_boarding_count", "sum"), Bs=("boarding_count", "sum"),
                                      days=("service_date", "nunique"))
allstops = df.groupby("source_stop_id").size().index
g = g.reindex(allstops)
diag = pd.DataFrame({"source_stop_id": g.index})
diag["average_daily_boarding"] = g.B.round(1).to_numpy()
diag["average_daily_alighting"] = g.A.round(1).to_numpy()
diag["city_boarding_rank"] = pd.array(g.B.rank(ascending=False, method="min").to_numpy(), dtype="Int64")
tot = (g.B + g.A).to_numpy()
br = np.where(tot > 0, g.B.to_numpy() / np.where(tot > 0, tot, 1), np.nan)
diag["boarding_ratio"] = np.round(br, 3)
diag["transfer_ratio"] = np.round(np.where(g.Bs > 0, g.Tb / g.Bs.where(g.Bs > 0, 1), np.nan), 3)
diag["stop_profile"] = np.select([br >= 0.6, br <= 0.4, ~np.isnan(br)], ["DEPARTURE", "ARRIVAL", "BALANCED"], None)
diag["observed_days"] = g.days.fillna(0).astype(int).to_numpy()
# 위험도: 마을버스 노선이 들어오기 전까지 미산정
diag["risk_level"] = "UNKNOWN"
for c in ("risk_reason", "serving_route_count", "passengers_per_route", "min_alternative_routes"):
    diag[c] = None
diag.to_csv(out / "stop_diagnosis.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------ 3. 분석 실행 기록
run = {
    "run_key": str(uuid.uuid4()),
    "algorithm_version": bundle.get("algorithm_version", "rgcn_v1"),
    "data_start": str(d0.date()), "data_end": str(d1.date()),
    "status": "PENDING  (백엔드가 모든 행을 넣은 뒤 COMPLETED로 변경)",
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "config": {"model_file": Path(a.model).name, "model_train_end": bundle.get("train_end"),
               "threshold": thr_cfg, "min_consecutive_days": MIN_DAYS, "merge_gap_days": MAX_GAP,
               "lag_correction_days": LAG, "stats_days": a.stats_days, "usage_definition": "boarding+alighting"},
    "counts": {"stop_diagnosis": len(diag), "usage_alert": len(alerts), "alert_daily_detail": len(detail)},
}
json.dump(run, open(out / "analysis_run.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(run["counts"], ensure_ascii=False))
a_df = pd.DataFrame(alerts, columns=alert_cols)
if len(a_df):
    print(a_df.groupby(["change_direction", "classification"]).size().to_string())
