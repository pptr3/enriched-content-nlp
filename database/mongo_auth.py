from pydantic_settings import BaseSettings


class MongoAuth(BaseSettings):
    """
    This class defines the settings for the application.
    It inherits from pydantic.BaseSettings and defines the required environment variables.
    """

    mongo_initdb_root_username: str
    mongo_initdb_root_password: str

    class Config:
        """
        This class defines the configuration for the Settings class.
        It specifies the environment file to use.
        """

        env_file = ".env"


mongo_auth = MongoAuth()
