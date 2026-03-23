import steelflow


def test_version_exposed() -> None:
    assert isinstance(steelflow.__version__, str)
    assert steelflow.__version__
