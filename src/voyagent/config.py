from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    google_places_api_key: str
    duffel_api_key: str = ""
    you_api_key: str = ""

    # SMTP — used only to email the finished itinerary to the traveler (the graph's second
    # human-approved write action). All optional: with no host/username/password set, that step
    # reports "not connected yet" instead of failing, exactly like the Duffel and Calendar tools.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""  # defaults to smtp_username when blank

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024

    chunk_size: int = 2000
    chunk_overlap: int = 300


settings = Settings()
