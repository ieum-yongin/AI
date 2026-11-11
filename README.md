# 이음 · AI

용인시 버스 노선 진단 서비스 **이음**의 AI 파트 저장소예요. 분석 결과를 백엔드 DB 테이블에 넣고, 시뮬레이션 함수를 제공해요.

## 하는 일

| 모듈 | 내용 | 결과가 들어가는 테이블 |
|---|---|---|
| `ieum_ai/forecast.py` | 정류소 이용량 7일 예측 (비교 기준 + GNN) | 급변 알림의 기준 이용량 |
| `ieum_ai/graph.py` | 정류소 그래프 (노선 연결 + 300m 거리) | GNN 입력 |
| `ieum_ai/dataset.py` | 일별 이용량 행렬, 달력 특성, 기간 나누기 | |
| `ieum_ai/simulation.py` | 노선 운행 중단 시뮬레이션 `simulate()` | `simulation_stop_result` (백엔드가 저장) |

## 테이블 규칙
- AI가 쓰는 테이블: `analysis_run`, `stop_diagnosis`, `usage_alert`, `alert_daily_detail`, `stop_hub_access`, `route_overlap`
- 나머지는 읽기만. 분석할 때마다 새 `analysis_run`으로 **추가만** 하고, 끝나면 `COMPLETED`로 바꿔요.
- `usage_alert.review_status`는 건드리지 않아요 (백엔드 전용).

## 데이터
- `data/stop_daily_usage.csv.gz`: 용인시 정류소별 일별 승하차, 2024-10-18 ~ 2026-08-31 (경기데이터드림 「경기도 정류소 별 승하차 인원 집계」)
- 2024-09-01 ~ 10-17은 정류소 약 30%가 빠진 날이 많아 제외했어요.
- 2025-12-09는 원본에 데이터가 없어요.

## 실행
```bash
pip install -r requirements.txt
python -m pytest tests            # 시뮬레이션 테스트
python scripts/run_forecast.py    # 예측 모델 학습·비교 → results/
```


## 백엔드 연동
전달 문서: [HANDOVER.md](HANDOVER.md) · 월간 배치: `python scripts/run_batch.py --from YYYY-MM-DD --to YYYY-MM-DD --out out/YYYY-MM`
