from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    """
    This class defines the settings for the application.
    It inherits from pydantic.BaseSettings and defines the required environment variables.
    """

    server_url: str
    realm: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str

    class Config:
        """
        This class defines the configuration for the Settings class.
        It specifies the environment file to use.
        """

        env_file = ".env"


auth_settings = AuthSettings()
