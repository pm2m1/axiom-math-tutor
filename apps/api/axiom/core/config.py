from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration, populated from environment variables.

    Kept intentionally small: add a field here only once something in the
    application actually reads it.
    """

    model_config = SettingsConfigDict(env_prefix="AXIOM_", env_file=".env", extra="ignore")

    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
