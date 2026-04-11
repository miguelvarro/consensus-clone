from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    data_dir: str = "./data"
    bm25_index_dir: str = "./data/indices/bm25"

    # none | openai | ollama
    llm_provider: str = "none"
    llm_api_key: str = ""

    # modelo y timeout
    llm_model: str = "qwen2.5:72b-instruct"
    llm_timeout: float = 120.0

    # para Ollama
    llm_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"
        case_sensitive = False
