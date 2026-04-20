from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    model_name: str = "iris-classifier"
    model_version: str = "latest"
    database_url: str = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/mlops_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()