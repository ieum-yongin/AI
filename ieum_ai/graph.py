"""이음 — 정류소 그래프 만들기 (GNN 입력용).

노드: 정류소
엣지 두 종류
  - route : 같은 노선·같은 방향에서 바로 이어지는 정류소 (정류소 순서 기준)
  - near  : 직선거리 radius_m 이내 정류소 (마을버스처럼 노선 데이터에 빠진 연결을 보완)

엣지는 양방향으로 넣고, 같은 쌍이 여러 번 나오면 한 번만 남긴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R = 6_371_000.0


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """정류소 쌍별 거리(m). 정류소 수가 수천 개 수준일 때 사용."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def route_edges(route_stops: pd.DataFrame) -> pd.DataFrame:
    """같은 노선·방향에서 순서상 바로 이어지는 정류소 쌍.

    route_stops 컬럼: route_id, direction, seq, stop_id
    """
    df = route_stops.sort_values(["route_id", "direction", "seq"])
    nxt = df.groupby(["route_id", "direction"])["stop_id"].shift(-1)
    e = pd.DataFrame({"src": df["stop_id"], "dst": nxt}).dropna()
    e = e[e.src != e.dst]
    return e.astype({"src": df["stop_id"].dtype, "dst": df["stop_id"].dtype})


def near_edges(stops: pd.DataFrame, radius_m: float = 300.0) -> pd.DataFrame:
    """좌표가 있는 정류소끼리 radius_m 이내 쌍. stops 컬럼: stop_id, lat, lon"""
    s = stops.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    d = haversine_matrix(s.lat.to_numpy(), s.lon.to_numpy())
    i, j = np.where((d <= radius_m) & (d > 0))
    return pd.DataFrame({"src": s.stop_id.to_numpy()[i], "dst": s.stop_id.to_numpy()[j],
                         "distance_m": d[i, j]})


def build_graph(stops: pd.DataFrame, route_stops: pd.DataFrame,
                radius_m: float = 300.0) -> tuple[list, np.ndarray, np.ndarray]:
    """GNN용 그래프.

    Returns
    -------
    node_ids : 정류소 ID 목록 (인덱스 = 노드 번호)
    edge_index : (2, E) int 배열, 양방향
    edge_type : (E,) int 배열, 0 = route, 1 = near
    """
    node_ids = list(stops.stop_id)
    idx = {s: i for i, s in enumerate(node_ids)}

    pairs = {}
    for t, e in ((0, route_edges(route_stops)), (1, near_edges(stops, radius_m))):
        for a, b in zip(e.src, e.dst):
            if a in idx and b in idx:
                for u, v in ((idx[a], idx[b]), (idx[b], idx[a])):
                    pairs.setdefault((u, v), t)   # route 연결을 우선 표시

    if not pairs:
        return node_ids, np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    uv = np.array(list(pairs.keys()), dtype=np.int64).T
    et = np.array(list(pairs.values()), dtype=np.int64)
    return node_ids, uv, et
