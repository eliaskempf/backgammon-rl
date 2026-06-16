"""Smoke test so the suite is non-empty and the package imports cleanly."""


def test_package_imports():
    import bgrl

    assert bgrl.__version__
