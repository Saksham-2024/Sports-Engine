import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.core.utils.configs import PathResolver

def test_path_resolver_init_success():
    """Test PathResolver successfully finds the config and project root."""
    resolver = PathResolver()
    assert resolver.project_root is not None
    assert (resolver.project_root / "configs" / "configs.yaml").exists()
    assert isinstance(resolver.configs, dict)

@patch("pathlib.Path.exists")
def test_path_resolver_init_fail(mock_exists):
    """Test PathResolver raises RuntimeError when no config is found."""
    mock_exists.return_value = False
    with pytest.raises(RuntimeError, match="Could not resolve project root"):
        PathResolver()

def test_get_path():
    """Test get_path returns a resolved Path object based on dot notation."""
    resolver = PathResolver()
    # Assume global.project_root exists in configs.yaml based on previous context
    # If not, let's use a mocked config
    resolver.configs = {"a": {"b": "path/to/dir"}}
    result = resolver.get_path("a.b")
    assert isinstance(result, Path)
    assert result == resolver.project_root / "path/to/dir"

def test_get_config_with_key():
    """Test get_config returns correct value using dot notation."""
    resolver = PathResolver()
    resolver.configs = {"global": {"random_seed": 42}}
    assert resolver.get_config("global.random_seed") == 42

def test_get_config_without_key():
    """Test get_config returns full dictionary when no key is provided."""
    resolver = PathResolver()
    resolver.configs = {"global": {"random_seed": 42}}
    assert resolver.get_config() == {"global": {"random_seed": 42}}
