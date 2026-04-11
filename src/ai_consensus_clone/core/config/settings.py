from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    data_dir: str = "./data"
    bm25_index_dir: str = "./data/indices/bm25"

    llm_provider: str = "none"   # none | openai
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_timeout: float = 45.0

    class Config:
        env_file = ".env"
