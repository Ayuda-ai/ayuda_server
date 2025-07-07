from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # DB Fields (parsed from .env)
    postgres_db: str
    postgres_user: str
    postgres_password: str
    database_host: str
    database_port: int

    # JWT Configs
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15

    # Pinecone Configs
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index: str

    # Redis Configs
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Neo4j Configs (optional)
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None

    # Ollama Configs
    ollama_address: str = "http://localhost:11434/api/generate"
    ollama_model: str = "ayuda_llama3"

    # Dynamically build DATABASE_URL from individual components
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.database_host}:{self.database_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False  # This makes environment variables case-insensitive
        extra = "ignore"  # Ignore extra fields

settings = Settings()
