from ieum_ai.simulation import (
    CAUTION, DANGER, SAFE, Network, RiskCriteria, classify_risk, simulate,
)


def make_network():
    # 정류소 1: 노선 A만 (승차 많음)   → 원래 DANGER
    # 정류소 2: 노선 A, B (승차 적음)   → CAUTION
    # 정류소 3: 노선 A, B, C (승차 50)  → SAFE
    # 정류소 4: 노선 C만 (승차 10)      → CAUTION, A 중단과 무관
    return Network(
        network_id=1,
        route_stops={10: [1, 2, 3], 20: [2, 3], 30: [3, 4]},
        daily_boarding={1: 500, 2: 40, 3: 50, 4: 10},
        is_complete=False,
    )


def test_classify_risk():
    c = RiskCriteria()
    assert classify_risk(0, 0, c) == DANGER
    assert classify_risk(1, 500, c) == DANGER
    assert classify_risk(1, 10, c) == CAUTION
    assert classify_risk(2, 10, c) == CAUTION
    assert classify_risk(3, 50, c) == SAFE
    assert classify_risk(3, 1200, c) == CAUTION  # 노선당 400명


def test_route_off_changes_only_its_stops():
    result = simulate(make_network(), [10])
    ids = {s["stop_id"] for s in result["stops"]}
    assert ids == {1, 2, 3}                      # 정류소 4는 무관

    by_id = {s["stop_id"]: s for s in result["stops"]}
    assert by_id[1]["is_isolated"] is True
    assert by_id[1]["remaining_route_count"] == 0
    assert by_id[1]["estimated_affected_daily"] == 500
    assert by_id[2]["after_risk_level"] == CAUTION  # B 하나 남음, 승차 40
    assert by_id[3]["after_risk_level"] == CAUTION  # 2개 남음


def test_summary_counts():
    result = simulate(make_network(), [10])
    s = result["summary"]
    assert s["isolated_stop_count"] == 1
    assert s["danger_delta"] == 0        # 정류소1 DANGER→DANGER
    assert s["caution_delta"] == 1       # 정류소3 SAFE→CAUTION
    # 500*1/1 + 40*1/2 + 50*1/3 = 536.7
    assert s["affected_riders"] == 537
    assert result["is_provisional"] is True


def test_unknown_routes_are_reported_not_raised():
    result = simulate(make_network(), [10, 999])
    assert result["unknown_route_ids"] == [999]
    assert result["summary"]["routes_off"] == [10]


def test_empty_request():
    result = simulate(make_network(), [])
    assert result["stops"] == []
    assert result["summary"]["affected_riders"] == 0


def test_build_network_from_rows():
    from ieum_ai.simulation import build_network
    net = build_network(7, [(10, 1), (10, 2), (20, 2), (10, 2)], [(1, 500), (2, None)])
    assert net.routes_at(2) == {10, 20}
    assert net.daily_boarding == {1: 500.0}
    assert simulate(net, [10])["summary"]["isolated_stop_count"] == 1
