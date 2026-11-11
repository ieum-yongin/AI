"""테스트 기간(2026-06-02 ~ 08-31) 급변 알림을 기준선별로 비교.

기준선: 'seasonal'(같은 요일 최근 4주 중앙값) vs 'gnn'(GNN 1일 앞 예측)
규칙: 실제/기준 비율의 7일 이동 중앙값이 0.6 미만(급감) 또는 1.6 초과(급증)인 정류소
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieum_ai import forecast as F

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
names = df.drop_duplicates("source_stop_id").set_index("source_stop_id").stop_name
p = F.prepare(df)
test = F.origins(p, "test")

base = {
    "seasonal": np.expm1(F.seasonal_naive(p, test)[:, 0]),            # (D, N) 1일 앞
    "gnn": np.expm1(np.load(ROOT / "results/gnn_test_pred.npy")[:, 0]),
}
actual = p.raw[test]
obs = p.obs[test]
stop_ids = np.array(p.stop_ids)

rows = []
for name, b in base.items():
    ratio = np.where(obs & (b > 0), actual / np.maximum(b, 1e-6), np.nan)
    r7 = pd.DataFrame(ratio).rolling(7, min_periods=5).median().to_numpy()
    flag = (r7 < 0.6) | (r7 > 1.6)
    hit = np.where(flag.any(axis=0))[0]
    err = np.nanmean(np.abs(np.log(np.where(ratio > 0, ratio, np.nan))))
    for j in hit:
        rows.append({"baseline": name, "stop_id": stop_ids[j], "stop_name": names.get(stop_ids[j], ""),
                     "direction": "DECREASE" if np.nanmin(r7[:, j]) < 0.6 else "INCREASE",
                     "days_flagged": int(flag[:, j].sum())})
    print(f"{name}: 알림 정류소 {len(hit)}곳, 평균 |log 비율| {err:.3f}")

out = pd.DataFrame(rows)
school = out.stop_name.str.contains("학교|고등|중학|초등|대학|단국대|외대|명지대|용인대|강남대|경희대|사색의광장|생명과학대")
out["school_like"] = school
print(out.groupby("baseline").agg(정류소=("stop_id", "count"), 학교_관련=("school_like", "sum")))
s_only = set(out[out.baseline == "seasonal"].stop_id) - set(out[out.baseline == "gnn"].stop_id)
g_only = set(out[out.baseline == "gnn"].stop_id) - set(out[out.baseline == "seasonal"].stop_id)
print("seasonal에만:", len(s_only), "| gnn에만:", len(g_only))
out.to_csv(ROOT / "results/alert_compare_test.csv", index=False, encoding="utf-8-sig")
