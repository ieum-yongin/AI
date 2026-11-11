"""이음 — 정류소 이용량 7일 예측: 비교 기준 모델과 GNN.

문제 정의
    과거 28일의 정류소별 하루 이용량(승차+하차)으로 다음 7일을 예측한다.
    예측이 정확할수록, 실제가 예측에서 크게 벗어난 날을 '급변'으로 볼 수 있다.

모델
    seasonal : 최근 4주 같은 요일의 중앙값 (지금 급변 알림 방식)
    lgbm     : LightGBM, 정류소마다 따로가 아니라 전체를 한 모델로 (그래프 없음)
    gru      : 정류소별 GRU (그래프 없음)
    gnn      : 정류소별 GRU + 그래프 합성곱(GCN) 2층  ← gru와 같은 구조에 그래프만 추가

평가
    테스트 기간(마지막 91일) 동안 7일 예측의 MAE, WAPE (원래 단위, 관측된 값만)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

IN_LEN, OUT_LEN = 28, 7


# ---------------------------------------------------------------- 데이터
@dataclass
class Prepared:
    dates: pd.DatetimeIndex
    stop_ids: list
    y: np.ndarray          # (T, N) log1p(이용량), 결측은 채운 값
    obs: np.ndarray        # (T, N) bool, 실제 관측 여부
    raw: np.ndarray        # (T, N) 원래 이용량 (결측 NaN)
    cal: np.ndarray        # (T, 8) 요일 원핫 + 전체 급감일
    node_mean: np.ndarray  # (N,) 학습 구간 log 평균
    split: tuple           # (train_end, val_end) 날짜 인덱스


def prepare(df: pd.DataFrame, min_daily_usage=10.0, min_coverage=0.9,
            val_days=91, test_days=91) -> Prepared:
    df = df.assign(usage=df.boarding_count + df.alighting_count)
    dates = pd.date_range(df.service_date.min(), df.service_date.max(), freq="D")
    u = df.pivot_table(index="service_date", columns="source_stop_id", values="usage", aggfunc="sum").reindex(dates)
    keep = (u.notna().mean() >= min_coverage) & (u.mean() >= min_daily_usage)
    u = u.loc[:, keep]
    raw = u.to_numpy(float)
    obs = ~np.isnan(raw)
    T = len(dates)
    train_end, val_end = T - test_days - val_days, T - test_days

    # 전체 급감일(공휴일·명절·폭설): 용인 전체가 같은 요일 중앙값의 80% 미만
    total = pd.Series(np.nansum(raw, axis=1), index=dates)
    off = (total / total.groupby(dates.dayofweek).transform("median") < 0.8).to_numpy()
    cal = np.concatenate([np.eye(7)[dates.dayofweek], off[:, None]], axis=1).astype(np.float32)

    # 결측 채우기: 같은 정류소·같은 요일의 학습 구간 중앙값
    y = np.log1p(raw)
    fill = np.zeros_like(y)
    for d in range(7):
        rows = np.where((dates.dayofweek == d)[:train_end])[0]
        med = np.nanmedian(y[rows], axis=0)
        fill[dates.dayofweek == d] = med
    y = np.where(obs, y, fill)
    node_mean = y[:train_end].mean(axis=0)
    return Prepared(dates, list(u.columns), y.astype(np.float32), obs, raw, cal, node_mean.astype(np.float32),
                    (train_end, val_end))


def origins(p: Prepared, part: str, step=1) -> np.ndarray:
    """예측 시작점 t: 입력 [t-IN_LEN, t), 정답 [t, t+OUT_LEN). 정답이 그 구간 안에 있어야 함."""
    tr, va = p.split
    T = len(p.dates)
    lo, hi = {"train": (IN_LEN, tr - OUT_LEN), "val": (tr, va - OUT_LEN), "test": (va, T - OUT_LEN)}[part]
    return np.arange(lo, hi + 1, step)


def score(p: Prepared, ts: np.ndarray, pred_log: np.ndarray) -> dict:
    """pred_log: (len(ts), OUT_LEN, N). 원래 단위로 MAE, WAPE."""
    err = tot = 0.0
    n = 0
    for i, t in enumerate(ts):
        true = p.raw[t:t + OUT_LEN]
        m = p.obs[t:t + OUT_LEN]
        pred = np.expm1(pred_log[i])
        e = np.abs(pred - np.nan_to_num(true))[m]
        err += e.sum(); tot += np.nan_to_num(true)[m].sum(); n += m.sum()
    return {"MAE": err / n, "WAPE": err / tot}


# ---------------------------------------------------------------- 비교 기준 1
def seasonal_naive(p: Prepared, ts: np.ndarray) -> np.ndarray:
    out = np.empty((len(ts), OUT_LEN, p.y.shape[1]), np.float32)
    for i, t in enumerate(ts):
        for h in range(OUT_LEN):
            past = [t + h - 7 * k for k in range(1, 5)]   # 같은 요일 지난 4주 (모두 입력 구간 안)
            out[i, h] = np.median(p.y[past], axis=0)
    return out


# ---------------------------------------------------------------- 비교 기준 2
def _lgbm_rows(p: Prepared, ts: np.ndarray, with_target=True):
    N = p.y.shape[1]
    X, Y, W = [], [], []
    for t in ts:
        win = p.y[t - IN_LEN:t]                           # (28, N)
        last7 = win[-7:].T                                 # (N, 7)
        stats = np.stack([win.mean(0), win.std(0), win[-7:].mean(0)], 1)
        for h in range(OUT_LEN):
            same = np.stack([p.y[t + h - 7 * k] for k in range(1, 5)], 1)   # (N, 4)
            calh = np.repeat(p.cal[t + h][None], N, 0)
            hcol = np.full((N, 1), h, np.float32)
            X.append(np.hstack([same, last7, stats, calh, hcol, p.node_mean[:, None]]))
            if with_target:
                Y.append(p.y[t + h]); W.append(p.obs[t + h])
    X = np.vstack(X).astype(np.float32)
    return (X, np.concatenate(Y), np.concatenate(W)) if with_target else X


def lgbm_model(p: Prepared, seed=0):
    import lightgbm as lgb
    tr = origins(p, "train", step=4)
    X, Y, W = _lgbm_rows(p, tr)
    Xv, Yv, Wv = _lgbm_rows(p, origins(p, "val", step=4))
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(X[W], Y[W], eval_set=[(Xv[Wv], Yv[Wv])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def lgbm_predict(m, p: Prepared, ts: np.ndarray) -> np.ndarray:
    N = p.y.shape[1]
    out = m.predict(_lgbm_rows(p, ts, with_target=False)).astype(np.float32)
    return out.reshape(len(ts), OUT_LEN, N)


# ---------------------------------------------------------------- 신경망 (GRU / GNN)
def normalized_adj(edge_index: np.ndarray, n: int) -> torch.Tensor:
    """D^-1/2 (A+I) D^-1/2, 희소 행렬."""
    ei = np.concatenate([edge_index, np.vstack([np.arange(n), np.arange(n)])], axis=1)
    deg = np.bincount(ei[0], minlength=n).astype(np.float32)
    w = 1.0 / np.sqrt(deg[ei[0]] * deg[ei[1]])
    return torch.sparse_coo_tensor(torch.tensor(ei), torch.tensor(w), (n, n)).coalesce()


class Forecaster(nn.Module):
    """정류소별 GRU 인코더 → (선택) GCN 2층 → 7일 출력.

    use_graph=False면 GRU 모델, True면 GNN. 나머지 구조가 같아서 그래프의 효과만 비교된다.
    adj에 인접행렬 하나를 주면 GCN, 여러 개(예: [노선 연결, 거리 연결])를 주면
    관계별로 따로 가중치를 배우는 관계형 GCN(R-GCN)이 된다.
    """

    def __init__(self, n_nodes: int, cal_dim=8, hidden=48, use_graph=True, adj=None):
        super().__init__()
        self.use_graph = use_graph
        self.adjs = adj if isinstance(adj, (list, tuple)) else ([adj] if adj is not None else [])
        self.node_emb = nn.Embedding(n_nodes, 8)
        self.gru = nn.GRU(1 + cal_dim + 8, hidden, batch_first=True)
        if use_graph:
            self.layers = nn.ModuleList(
                nn.ModuleList(nn.Linear(hidden, hidden, bias=(k == 0)) for k in range(len(self.adjs)))
                for _ in range(2))
        self.head = nn.Sequential(nn.Linear(hidden + OUT_LEN * cal_dim, 64), nn.ReLU(), nn.Linear(64, OUT_LEN))

    def forward(self, x, cal_in, cal_out):
        # x: (B, N, IN_LEN)  cal_in: (B, IN_LEN, C)  cal_out: (B, OUT_LEN, C)
        B, N, L = x.shape
        emb = self.node_emb.weight[None, :, None, :].expand(B, N, L, -1)
        ci = cal_in[:, None].expand(B, N, L, -1)
        seq = torch.cat([x[..., None], ci, emb], -1).reshape(B * N, L, -1)
        _, h = self.gru(seq)
        h = h[-1].reshape(B, N, -1)
        if self.use_graph:
            for rel_lins in self.layers:
                msg = 0
                for A, lin in zip(self.adjs, rel_lins):   # 관계(연결 종류)마다 따로 변환
                    msg = msg + lin(torch.stack([torch.sparse.mm(A, h[b]) for b in range(B)]))
                h = h + torch.relu(msg)                   # 잔차 연결: 자기 정보 + 이웃 정보
        co = cal_out.reshape(B, 1, -1).expand(B, N, -1)
        return self.head(torch.cat([h, co], -1))   # (B, N, OUT_LEN) 정규화된 log 값


def _batch(p: Prepared, ts):
    mu = p.node_mean
    x = np.stack([(p.y[t - IN_LEN:t] - mu).T for t in ts])
    y = np.stack([(p.y[t:t + OUT_LEN] - mu).T for t in ts])
    m = np.stack([p.obs[t:t + OUT_LEN].T for t in ts])
    ci = np.stack([p.cal[t - IN_LEN:t] for t in ts])
    co = np.stack([p.cal[t:t + OUT_LEN] for t in ts])
    return [torch.tensor(a) for a in (x, y, m, ci, co)]


def train_nn(p: Prepared, use_graph: bool, adj=None, epochs=40, batch=8, lr=3e-3, seed=0, log=print,
             train_step=1):
    torch.manual_seed(seed); np.random.seed(seed)
    N = p.y.shape[1]
    model = Forecaster(N, use_graph=use_graph, adj=adj)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr, va = origins(p, "train", step=train_step), origins(p, "val", step=2)
    vb = _batch(p, va)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for idx in np.array_split(np.random.permutation(tr), max(1, len(tr) // batch)):
            x, y, m, ci, co = _batch(p, idx)
            loss = (torch.abs(model(x, ci, co) - y) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            x, y, m, ci, co = vb
            vl = ((torch.abs(model(x, ci, co) - y) * m).sum() / m.sum()).item()
        log(f"  ep{ep:02d} val_L1={vl:.4f} ({time.time()-t0:.0f}s)")
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    return model


def nn_predict(model, p: Prepared, ts) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for idx in np.array_split(ts, max(1, len(ts) // 8)):
            x, _, _, ci, co = _batch(p, idx)
            out.append(model(x, ci, co).numpy())
    pred = np.concatenate(out)                         # (B, N, OUT_LEN)
    return (pred + p.node_mean[None, :, None]).transpose(0, 2, 1)
