from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str
    perplexity_api_key: str

    model_config = SettingsConfigDict(env_file='.env', extra="ignore")
