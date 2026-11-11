"""이음 — 이용량 예측용 데이터셋.

stop_daily_usage(일별 정류소 승하차)를 (날짜 × 정류소) 행렬로 바꾸고,
달력 특성과 학습/검증/테스트 구간을 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UsageMatrix:
    dates: pd.DatetimeIndex     # (T,)
    stop_ids: list              # (N,)
    usage: np.ndarray           # (T, N) 승차+하차, 미관측은 NaN
    boarding: np.ndarray        # (T, N)
    observed: np.ndarray        # (T, N) bool


def to_matrix(df: pd.DataFrame, min_daily_usage: float = 10.0) -> UsageMatrix:
    """긴 형식(행 = 정류소·날짜)을 행렬로. 평균 이용이 너무 적은 정류소는 뺀다."""
    df = df.copy()
    df["usage"] = df["boarding_count"] + df["alighting_count"]
    dates = pd.date_range(df.service_date.min(), df.service_date.max(), freq="D")
    u = df.pivot_table(index="service_date", columns="source_stop_id", values="usage", aggfunc="sum").reindex(dates)
    b = df.pivot_table(index="service_date", columns="source_stop_id", values="boarding_count", aggfunc="sum").reindex(dates)
    keep = u.mean(skipna=True) >= min_daily_usage
    u, b = u.loc[:, keep], b.loc[:, keep]
    return UsageMatrix(dates=dates, stop_ids=list(u.columns), usage=u.to_numpy(float),
                       boarding=b.to_numpy(float), observed=~np.isnan(u.to_numpy(float)))


def citywide_offdays(m: UsageMatrix, threshold: float = 0.8) -> np.ndarray:
    """용인 전체 이용이 같은 요일 중앙값의 threshold배 미만인 날 (공휴일·명절 등).

    공휴일 목록을 직접 관리하지 않고 데이터로 찾는다. (T,) bool
    """
    total = pd.Series(np.nansum(m.usage, axis=1), index=m.dates)
    med = total.groupby(total.index.dayofweek).transform("median")
    return (total / med < threshold).to_numpy()


def calendar_features(m: UsageMatrix) -> np.ndarray:
    """(T, 8): 요일 원핫 7개 + 전체 이용 급감일(공휴일 등) 1개."""
    dow = np.eye(7)[m.dates.dayofweek]
    off = citywide_offdays(m)[:, None].astype(float)
    return np.concatenate([dow, off], axis=1)


def time_split(dates: pd.DatetimeIndex, val_days: int = 91, test_days: int = 91):
    """시간 순서대로 학습 / 검증 / 테스트 구간 인덱스 (미래 정보 누설 방지)."""
    T = len(dates)
    test0, val0 = T - test_days, T - test_days - val_days
    return np.arange(0, val0), np.arange(val0, test0), np.arange(test0, T)
