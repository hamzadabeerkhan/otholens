from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    allowed_origins: str = "http://localhost:3000"
    max_upload_bytes: int = 10 * 1024 * 1024
    enable_model: bool = False
    model_path: Path = Path("/app/artifacts/models/ortholens-kl-resnet18-v0.1.0.pt")
    minimum_confidence: float = 0.55

    model_config = SettingsConfigDict(env_prefix="ORTHOLENS_")

    @property
    def origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
