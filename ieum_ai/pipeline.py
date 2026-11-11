"""운영용 도구: 그래프 만들기, 모델 묶음(bundle) 저장·불러오기, 새 데이터를 모델 정류소 순서에 맞추기.

모델 묶음(.pt) 안에는 가중치와 함께 정류소 순서(stop_ids), 정류소별 평균(node_mean),
알림 임계값, 학습 기간이 들어 있다. 새 달 데이터가 와도 이 순서에 맞춰서 예측한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import forecast as F
from .graph import build_graph


def load_network(route_csv, stop_csv):
    """BMS 노선 경유 정류소 + 정류소 위치 CSV를 읽는다 (용인시 공공데이터 형식)."""
    r = pd.read_csv(route_csv, encoding="utf-8-sig", dtype={"노선ID": str, "정류소ID": str})
    s = pd.read_csv(stop_csv, encoding="utf-8-sig", dtype={"정류소ID": str})
    rs = r.rename(columns={"노선ID": "route_id", "진행구분코드": "direction", "정류소순서": "seq", "정류소ID": "stop_id"})
    loc = s.rename(columns={"정류소ID": "stop_id", "WGS84위도": "lat", "WGS84경도": "lon"})[["stop_id", "lat", "lon"]]
    return rs[["route_id", "direction", "seq", "stop_id"]], loc


def rgcn_adj(stop_ids, route_stops, stop_loc):
    stops = pd.DataFrame({"stop_id": list(stop_ids)}).merge(stop_loc, how="left")
    _, ei, et = build_graph(stops, route_stops)
    n = len(stop_ids)
    return [F.normalized_adj(ei[:, et == 0], n), F.normalized_adj(ei[:, et == 1], n)]


def save_bundle(path, model, p: F.Prepared, **meta):
    torch.save({"state_dict": model.state_dict(), "stop_ids": list(p.stop_ids),
                "node_mean": p.node_mean, **meta}, path)


def load_bundle(path, route_stops, stop_loc):
    b = torch.load(path, weights_only=False)
    m = F.Forecaster(len(b["stop_ids"]), use_graph=True, adj=rgcn_adj(b["stop_ids"], route_stops, stop_loc))
    m.load_state_dict(b["state_dict"])
    return m.eval(), b


def align(df: pd.DataFrame, stop_ids, node_mean) -> F.Prepared:
    """새 데이터를 모델의 정류소 순서에 맞춘다. 없는 정류소·날짜는 같은 요일 최근 12주 중앙값으로 채움."""
    df = df.assign(usage=df.boarding_count + df.alighting_count)
    dates = pd.date_range(df.service_date.min(), df.service_date.max(), freq="D")
    u = df.pivot_table(index="service_date", columns="source_stop_id", values="usage", aggfunc="sum")
    u = u.reindex(index=dates, columns=list(stop_ids))
    raw = u.to_numpy(float)
    obs = ~np.isnan(raw)
    total = pd.Series(np.nansum(raw, axis=1), index=dates)
    off = (total / total.groupby(dates.dayofweek).transform("median") < 0.8).to_numpy()
    cal = np.concatenate([np.eye(7)[dates.dayofweek], off[:, None]], axis=1).astype(np.float32)
    y = np.log1p(raw)
    fill = np.tile(node_mean, (len(dates), 1))
    for d in range(7):
        rows = np.where(dates.dayofweek == d)[0]
        med = np.nanmedian(y[rows[-12:]], axis=0)
        fill[rows] = np.where(np.isnan(med), node_mean, med)
    y = np.where(obs, y, fill).astype(np.float32)
    T = len(dates)
    return F.Prepared(dates, list(stop_ids), y, obs, raw, cal, np.asarray(node_mean, np.float32), (T, T))


def baseline_1day(model, p: F.Prepared, day_idx: np.ndarray) -> np.ndarray:
    """날짜 d의 기준선 = d 전날까지 28일로 만든 1일 앞 예측. (len(day_idx), N) 원래 단위."""
    k = F.OUT_LEN
    fut = pd.date_range(p.dates[-1] + pd.Timedelta(days=1), periods=k, freq="D")
    cal_f = np.concatenate([np.eye(7)[fut.dayofweek], np.zeros((k, 1))], axis=1).astype(np.float32)
    pp = F.Prepared(p.dates.append(fut), p.stop_ids, np.vstack([p.y, np.repeat(p.y[-1:], k, 0)]),
                    np.vstack([p.obs, np.zeros((k, p.y.shape[1]), bool)]), p.raw, np.vstack([p.cal, cal_f]),
                    p.node_mean, p.split)   # 마지막 날 뒤 7일은 입력 형태만 맞추는 빈칸 (예측에 안 쓰임)
    return np.expm1(F.nn_predict(model, pp, day_idx)[:, 0])


def alert_score(p: F.Prepared, day_idx, base):
    """실제/기준선 비율 → 7일 이동 중앙값의 log. 양수=증가, 음수=감소. (D, N)"""
    act = p.raw[day_idx]
    ratio = np.where(p.obs[day_idx] & (base > 0), act / np.maximum(base, 1e-6), np.nan)
    r7 = pd.DataFrame(ratio).rolling(7, min_periods=5).median().to_numpy()
    return np.log(np.where(r7 > 0, r7, np.nan)), ratio
