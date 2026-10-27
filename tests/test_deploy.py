import numpy as np

from leafscan.deploy import selective, split_calibration


def test_split_calibration_is_disjoint_and_complete():
    items = [{"path": f"{c}/{i}.jpg", "label": c} for c in "ab" for i in range(20)]
    train, calib = split_calibration(items, 0.15, seed=0)
    paths = lambda xs: {x["path"] for x in xs}  # noqa: E731
    assert not paths(train) & paths(calib)
    assert paths(train) | paths(calib) == paths(items)
    assert len(calib) == 6  # 3 per class


def test_selective():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    y = np.array([0, 1, 1])
    assert selective(probs, y, 0.7) == {"coverage": 2 / 3, "accuracy_when_answering": 1.0}
    assert selective(probs, y, 0.95)["accuracy_when_answering"] is None
