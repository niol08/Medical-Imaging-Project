from pydantic import BaseModel
import tomllib
from pathlib import Path

class UI(BaseModel):
    app_name: str = "Medimodal AI"
    org: str = "Research"
    accent: str = "default"
    enable_llm: bool = True

class Modalities(BaseModel):
    spectroscopy: bool = True
    fluoroscopy: bool = True
    pet: bool = True

class Settings(BaseModel):
    ui: UI = UI()
    modalities: Modalities = Modalities()

def _load() -> Settings:
    cfg = Path("settings.toml")
    if cfg.exists():
        with cfg.open("rb") as f:
            d = tomllib.load(f)
        return Settings(**d)
    return Settings()

settings = _load()
