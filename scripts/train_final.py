"""서비스용 최종 관계형 GCN(rgcn_v1) 학습.

전체 데이터를 쓰되, 마지막 35일은 학습을 멈출 시점을 정하는 데만 쓴다.
결과: models/rgcn_v1.pt (가중치 + 정류소 순서 + 평균값 + 학습 기간)
알림 임계값은 scripts/calibrate_threshold.py 가 같은 파일에 넣는다.
"""
import sys, time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieum_ai import forecast as F
from ieum_ai import pipeline as P

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/yongin")

df = pd.read_csv(ROOT / "data/stop_daily_usage.csv.gz", dtype={"source_stop_id": str}, parse_dates=["service_date"])
p = F.prepare(df, val_days=35, test_days=0)
rs, loc = P.load_network(DATA / "용인시_버스노선_경유정류소.csv", DATA / "용인시_버스정류소_현황.csv")
adj = P.rgcn_adj(p.stop_ids, rs, loc)

t = time.time()
model = F.train_nn(p, use_graph=True, adj=adj, seed=0, train_step=2)
(ROOT / "models").mkdir(exist_ok=True)
P.save_bundle(ROOT / "models/rgcn_v1.pt", model, p, algorithm_version="rgcn_v1",
              train_start=str(p.dates[0].date()), train_end=str(p.dates[-1].date()),
              n_stops=len(p.stop_ids))
print(f"saved models/rgcn_v1.pt ({time.time()-t:.0f}s)")
