import functools

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

# Temporary clusterfuck


class Settings(BaseSettings):
    """
    Settings, constants, etc..
    """

    huggingface_token: SecretStr = Field(default=None, env="HUGGINGFACE_TOKEN")
    wandb_token: SecretStr = Field(default=None, env="WANDB_TOKEN")

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=(".env", "env.prod", "../.env"),
        env_file_encoding="utf-8",
        frozen=True,  # Prevents any modifications to the settings object
        protected_namespaces=("settings_",),
    )


@functools.lru_cache()
def get_settings() -> Settings:
    return Settings()
