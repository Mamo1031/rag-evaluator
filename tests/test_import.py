"""Minimal import tests to verify the package structure."""


def test_utils_import() -> None:
    """Utils module can be imported."""
    from src.utils import ChatClient, load_config

    assert ChatClient is not None
    assert load_config is not None
