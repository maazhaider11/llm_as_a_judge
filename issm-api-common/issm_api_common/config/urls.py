from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_db: str = Field(..., alias="MONGO_DB_NAME")
    database_host: str = Field(..., alias="MONGO_DATABASE_HOST")
    database_username: str = Field(..., alias="MONGO_INITDB_ROOT_USERNAME")
    database_password: str = Field(..., alias="MONGO_INITDB_ROOT_PASSWORD")
    database_port: int = Field(27017, alias="MONGO_DATABASE_PORT")

    neo4j_uri: str = Field("bolt://neo4j:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("password", alias="NEO4J_PASSWORD")

    @property
    def _database_url(self):
        return f"{self.database_host}:{self.database_port}"

    @property
    def mongo_database_conn_str(self) -> str:
        return f"mongodb://{self.database_username}:{self.database_password}@{self._database_url}/{self.database_db}?authSource=admin"


config = Settings()
