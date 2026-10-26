from tavp.dgp.processes import make_dgp
from tavp.tasks.strategies import build_strategies
from tavp.utils import REPO_ROOT, load_config

PREREG = load_config(REPO_ROOT / "configs" / "preregistration.yaml")


def test_processes_build_with_matched_volatility():
    for spec in PREREG["dgps"].values():
        d = make_dgp(spec)
        assert len(d.hash()) == 12


def test_grids_and_tolerances_complete():
    s = build_strategies(PREREG["use"]["strategies"])
    assert [len(s[k].grid) for k in ["BH", "TF", "VT", "TVF"]] == [0, 27, 18, 32]
    tol = PREREG["use"]["tolerances"]
    for metric, funcs in PREREG["use"]["functionals"].items():
        assert set(funcs) == set(tol[metric])
    assert set(PREREG["hypotheses"]) == {f"H{i}" for i in range(1, 8)}
