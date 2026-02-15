from gsrd.risk.control import clopper_pearson_ucb, enforce_monotone_nonincreasing, hoeffding_ucb


def test_monotone_nonincreasing_enforcement() -> None:
    risks = [0.5, 0.4, 0.45, 0.2]
    fixed = enforce_monotone_nonincreasing(risks)
    assert fixed[0] >= fixed[1] >= fixed[2] >= fixed[3]
    assert fixed == [0.5, 0.45, 0.45, 0.2]


def test_hoeffding_ucb_bounds() -> None:
    assert hoeffding_ucb(0.2, 100, 0.05) >= 0.2
    assert 0 <= hoeffding_ucb(0.0, 1000, 0.05) <= 1.0
    assert hoeffding_ucb(0.5, 0, 0.05) == 1.0


def test_clopper_pearson_ucb_bounds() -> None:
    assert 0 <= clopper_pearson_ucb(0, 10, 0.05) <= 1.0
    assert clopper_pearson_ucb(5, 10, 0.05) >= 0.5
    assert clopper_pearson_ucb(10, 10, 0.05) == 1.0
