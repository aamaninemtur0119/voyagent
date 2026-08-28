from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    google_places_api_key: str
    amadeus_api_key: str = ""
    amadeus_api_secret: str = ""

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024

    chunk_size: int = 2000
    chunk_overlap: int = 300


settings = Settings()
