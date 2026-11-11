# 이음 AI → 백엔드 전달 문서

AI 파트가 백엔드에 넘기는 것은 **두 가지**예요.

1. **분석 결과 CSV**: 매달 배치로 만들어요. 백엔드는 DB에 넣기만 하면 돼요.
2. **`simulate()` 함수**: 시뮬레이션 API가 요청마다 호출해요.

## 1. 분석 결과 CSV (승열 님)

데모용 결과는 `out/demo_2026-06_08/`에 있어요. 2026년 6~8월 분석이에요.

| 파일 | 넣을 테이블 | 행 수 | 비고 |
|---|---|---|---|
| `analysis_run.json` | analysis_run | 1 | config에 모델, 임계값, 설정이 들어 있음 |
| `stop_diagnosis.csv` | stop_diagnosis | 2,781 (모든 정류소) | 위험도 관련 값은 아래 설명 참고 |
| `usage_alert.csv` | usage_alert | 62 | 급감 42건, 급증 20건 |
| `alert_daily_detail.csv` | alert_daily_detail | 1,226 | `alert_key`로 usage_alert와 연결 |

**넣는 순서**

1. analysis_run 생성 (status = PENDING)
2. stop_diagnosis
3. usage_alert
4. alert_daily_detail
5. analysis_run.status를 COMPLETED로 변경

**ID 변환**

- 모든 파일의 정류소는 `source_stop_id`(경기도 정류소ID, 예: 228000159)로 되어 있어요.
- 백엔드에서 `bus_stop`의 stop_id로 바꿔서 넣어 주세요.
- `alert_daily_detail`은 `alert_key`로 usage_alert의 id를 찾아 넣으면 돼요.

**컬럼**

- **usage_alert**
  - `alert_key`: source_stop_id + change_start. 분석을 다시 돌려도 같은 알림을 알아보는 키예요. 이전 검토 기록을 이어 붙일 때 써요.
  - 그 외 컬럼: change_start, change_end, change_direction (INCREASE/DECREASE), classification (NEEDS_CHECK/EXPECTED), expected_reason, usage_ratio, consecutive_days, baseline_daily_usage, observed_daily_usage, analysis_summary
  - review_status는 없어요. 백엔드 기본값 NEW를 써 주세요.
- **alert_daily_detail**
  - 컬럼: alert_key, service_date, baseline_usage, observed_usage, is_anomalous
  - observed_usage는 미수집이면 빈칸(null)이에요.
- **stop_diagnosis**
  - 이용 관련 컬럼: average_daily_boarding, average_daily_alighting, city_boarding_rank, boarding_ratio, transfer_ratio, stop_profile, observed_days
    - 최근 91일 기준이에요.
    - 그 기간에 데이터가 없는 정류소는 빈칸이에요.
  - 위험도 관련 컬럼: risk_level은 `UNKNOWN`이고, risk_reason, serving_route_count, passengers_per_route, min_alternative_routes는 빈칸이에요.
    - 마을버스 노선이 들어오면 채울게요.

**값의 의미**

- **기준값 (baseline)**
  - 예전처럼 "같은 요일 평소값"이 아니라, AI 모델(관계형 GNN)이 **전날까지 데이터로 예측한 그날 이용량**이에요.
  - API 명세의 "처리 기준" 문구를 "AI가 예측한 평소 이용량"으로 바꿔 주세요.
- **알림 기준**: 실제/예측 비율이 **0.72배 이하** 또는 **1.39배 이상**인 상태가 3일 이상 이어지면 알림이에요.
  - 이 값은 평범한 정류소의 5%만 3개월 동안 오탐이 나도록 맞췄어요.
- **이용량**: 승차 + 하차예요.
- **EXPECTED 분류**: 지금은 공휴일·명절만 EXPECTED로 분류해요. 방학 분류는 calendar_event 테이블이 생기면 추가할게요.

## 2. simulate() (은비 님)

- **위치**: `ieum_ai/simulation.py`. 외부 라이브러리 없이 순수 파이썬이라 Django에서 바로 import할 수 있어요.

**사용법**

```python
from ieum_ai.simulation import build_network, simulate

# 서버 시작 시 한 번 (노선망이 바뀔 때만 다시)
net = build_network(
    network_id,
    RouteStop.objects.filter(pattern__network_id=nid).values_list("pattern__route_id", "stop_id"),
    StopDiagnosis.objects.filter(analysis_id=aid).values_list("stop_id", "average_daily_boarding"),
    is_complete=False,   # 마을버스 포함 노선망이면 True
)

# POST /api/simulations/ 요청마다 (1ms 미만)
result = simulate(net, routes_off=[route_id, ...])
```

**결과**

- `result["summary"]` → simulations/{id}/ 의 요약
  - danger_delta, caution_delta, affected_riders, isolated_stop_count, affected_stop_count, changed_stop_count
- `result["stops"]` → simulations/{id}/stops/ 의 정류소별 결과 (simulation_stop_result 한 행씩)
  - before_risk_level, after_risk_level, is_changed, before_route_count, remaining_route_count, after_min_alternatives, estimated_affected_daily, is_isolated, reason
- `result["is_provisional"]`
  - 마을버스가 빠진 노선망이면 true예요.
  - 화면에 "임시 결과" 표시용이에요.
- `result["criteria"]`: simulation.parameters_json에 저장해 주세요.
- `result["unknown_route_ids"]`: 노선망에 없는 노선 ID예요. 400 응답에 쓸 수 있어요.

**실제 데이터 예시**

- 파일: `examples/simulate_example.json`
- 노선 228000281을 운행 중단한 경우예요.
  - 영향 정류소 90곳, 위험 +21, 고립 15곳, 영향 승객 하루 약 1,651명
- API 명세의 응답 필드 이름을 이 키 그대로 쓰면 변환 없이 연결돼요. 이름을 바꾸고 싶으면 말씀해 주세요. 제가 맞출게요.

## 3. 모델 파일

| 파일 | 내용 |
|---|---|
| `models/rgcn_v1.pt` | 서비스용 모델. 2024-10 ~ 2026-08 데이터 사용 (7월까지 학습, 8월은 학습 종료 시점 결정용) |
| `models/rgcn_v1_holdout.pt` | 2026-03까지만 학습한 모델. 데모 결과(6~8월)는 이 모델로 만들었어요. 학습에 안 쓴 기간이라 실제 운영과 같은 조건이에요. |
| `models/threshold.json` | 알림 임계값 (목표 오탐 5%) |

**매달 돌리는 법**

```bash
python scripts/run_batch.py --from 2026-09-01 --to 2026-09-30 --out out/2026-09
```

- 실행 시간은 약 1분이에요. 9월 데이터부터는 `rgcn_v1.pt`를 써요.
