from pathlib import Path
from typing import Any, Union
import yaml

class PathResolver:
    def __init__(self):
        self.current = Path(__file__).resolve()
        for parent in list(self.current.parents)[:5]:
            self.config_path = parent / "configs" / "configs.yaml"
            if (self.config_path).exists():
                self.project_root = parent
                with open(self.config_path, "r") as f:
                    self.configs = yaml.safe_load(f)
                return
        raise RuntimeError(" Could not resolve project root")
    
    def get_path(self, key):
        keys = key.split(".")
        value = self.configs
        for k in keys:
            value = value[k]
        return self.project_root / value
    
    def get_config(self, key: str = None):
        """Returns any config value, dictionary, or parameter using dot notation."""
        if key is None:
            return self.configs
        value = self.configs
        for k in key.split("."):
            value = value[k]
        return value

        


