import functools

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """
    Settings, constants, etc..
    """

    huggingface_token: SecretStr = Field(default=None, env="HUGGINGFACE_TOKEN")


@functools.lru_cache()
def get_settings() -> Settings:
    return Settings()
