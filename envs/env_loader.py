"""Module for loading environment variables using dotenv."""

import os

from dotenv import load_dotenv


class EnvLoader:
    """Environment variable loader for the application."""

    def __init__(self):
        """Initialize the EnvLoader and load environment variables."""
        # Determine the environment type
        self._env_type = os.getenv("ENV", "development")
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_file_path = os.path.join(
            parent_path, ".env_vars", ".env.{}".format(self._env_type)
        )
        # Load the environment variables from the specified file
        load_dotenv(self.env_file_path)

    @property
    def env_type(self) -> str:
        """Return the environment type."""
        return self._env_type

    @property
    def secret_key(self):
        """Return the secret key."""
        return os.getenv("SECRET_KEY")

    @property
    def debug(self):
        """Return True if the environment is development, otherwise False."""
        return os.getenv("ENV_TYPE") == "development"

    @property
    def database_name(self):
        """Return the database name."""
        return os.getenv("DATABASE_NAME", "postgres")

    @property
    def database_user(self):
        """Return the database user."""
        return os.getenv("DATABASE_USER", "postgres")

    @property
    def database_password(self):
        """Return the database password."""
        return os.getenv("DATABASE_PASSWORD", "postgres")

    @property
    def database_host(self):
        """Return the database host."""
        return os.getenv("DATABASE_HOST", "localhost")

    @property
    def database_port(self):
        """Return the database port."""
        return os.getenv("DATABASE_PORT", "5432")

    @property
    def google_client_id(self):
        """Return the Google client ID."""
        return os.getenv("GOOGLE_CLIENT_ID")

    @property
    def google_client_secret(self):
        """Return the Google client secret."""
        return os.getenv("GOOGLE_SECRET")

    @property
    def google_redirect_uri(self):
        """Return the Google redirect URI."""
        return os.getenv("GOOGLE_REDIRECT_URI")

    @property
    def linkedin_client_id(self):
        """Return the LinkedIn client ID."""
        return os.getenv("LINKEDIN_CLIENT_ID")

    @property
    def linkedin_client_secret(self):
        """Return the LinkedIn client secret."""
        return os.getenv("LINKEDIN_SECRET")

    @property
    def linkedin_redirect_uri(self):
        """Return the LinkedIn redirect URI."""
        return os.getenv("LINKEDIN_REDIRECT_URI")

    @property
    def github_client_id(self):
        """Return the GitHub client ID."""
        return os.getenv("GITHUB_CLIENT_ID")

    @property
    def github_client_secret(self):
        """Return the GitHub client secret."""
        return os.getenv("GITHUB_SECRET")

    @property
    def github_redirect_uri(self):
        """Return the GitHub redirect URI."""
        return os.getenv("GITHUB_REDIRECT_URI")

    @property
    def jwt_authentication_timeout(self):
        """Return the JWT authentication timeout."""
        timeout_str = os.getenv("JWT_AUTHENTICATION_TIMEOUT", "3600")
        return int(timeout_str)

    @property
    def email_host(self):
        """Return the email host."""
        return os.getenv("EMAIL_HOST")

    @property
    def email_host_user(self):
        """Return the email host user."""
        return os.getenv("EMAIL_HOST_USER")

    @property
    def email_host_password(self):
        """Return the email host password."""
        return os.getenv("EMAIL_HOST_PASSWORD")

    @property
    def email_port(self):
        """Return the email port."""
        return os.getenv("EMAIL_PORT")

    @property
    def email_use_tls(self):
        """Return True if TLS is used for email, otherwise False."""
        return os.getenv("EMAIL_USE_TLS")

    @property
    def email_use_ssl(self):
        """Return True if SSL is used for email, otherwise False."""
        return os.getenv("EMAIL_USE_SSL")

    @property
    def openai_api_key(self):
        """Return the OpenAI API key."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def lancedb_path(self):
        """Return the LanceDB path."""
        return os.getenv("LANCEDB_PATH")

    @property
    def openai_model(self):
        """Return the OpenAI model."""
        return os.getenv("OPENAI_MODEL")

    @property
    def lancedb_path_counsaltant(self):
        """Return the LanceDB path for consultants."""
        return os.getenv("LANCEDB_PATH_COUNSALTANT")

    @property
    def pinecone_api_key(self):
        """Return the Pinecone API key."""
        return os.getenv("PINECONE_API_KEY")

    @property
    def pinecone_environment(self):
        """Return the Pinecone environment."""
        return os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

    @property
    def guest_user_query_url(self):
        """Return the guest user query URL."""
        return os.getenv("GUEST_USER_QUERY_URL")

    @property
    def django_settings_module(self):
        """Return the Django settings module."""
        return os.getenv("DJANGO_SETTINGS_MODULE")

    @property
    def django_auth_url(self):
        """Return the Django authentication URL."""
        return os.getenv("DJANGO_AUTH_URL")

    @property
    def verify_url(self):
        """Return the verification URL."""
        return os.getenv("VERIFY_URL")

    @property
    def stripe_secret_key(self):
        """Return the Stripe secret key."""
        return os.getenv("STRIPE_SECRET_KEY")

    @property
    def stripe_webhook_secret(self):
        """Return the Stripe webhook secret."""
        return os.getenv("STRIPE_TEST_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")

    @property
    def stripe_publishable_key(self):
        """Return the Stripe publishable key."""
        return os.getenv("STRIPE_PUBLISHABLE_KEY")

    @property
    def smtp_server(self):
        """Return the SMTP server."""
        return os.getenv("SMTP_SERVER")

    @property
    def smtp_port(self):
        """Return the SMTP port."""
        return os.getenv("SMTP_PORT")

    @property
    def email_address(self):
        """Return the email address."""
        return os.getenv("EMAIL_ADDRESS")

    @property
    def email_password(self):
        """Return the email password."""
        return os.getenv("EMAIL_PASSWORD")

    @property
    def mailgun_api_key(self):
        """Return the Mailgun API key."""
        return os.getenv("MAILGUN_API_KEY")

    @property
    def mailgun_domain(self):
        """Return the Mailgun domain."""
        return os.getenv("MAILGUN_DOMAIN")

    @property
    def rewardful_api_key(self):
        """Return the Rewardful API key."""
        return os.getenv("REWARDFUL_API_KEY")

    @property
    def rewardful_api_secret(self):
        """Return the Rewardful API secret."""
        return os.getenv("REWARDFUL_API_SECRET")

    @property
    def frontend_url(self):
        """Return the frontend URL."""
        return os.getenv("FRONTEND_URL")

    @property
    def base_url(self):
        """Return the base URL."""
        return os.getenv("BASE_URL")

    @property
    def change_password_url(self):
        """Return the change password URL."""
        return os.getenv("CHANGE_PASSWORD_URL")

    @property
    def password_reset_url(self):
        """Return the password reset URL."""
        return os.getenv("PASSWORD_RESET_URL")

    @property
    def fastapi_service_url(self):
        """Return the FastAPI service URL."""
        return os.getenv("FASTAPI_SERVICE_URL")

    @property
    def pinecone_contractor_index(self):
        """Return the Pinecone contractor index."""
        return os.getenv("PINECONE_CONTRACTOR_INDEX")

    @property
    def pinecone_tool_index(self):
        """Return the Pinecone tool index."""
        return os.getenv("PINECONE_TOOL_INDEX")

    @property
    def pinecone_workflow_index(self):
        """Return the Pinecone workflow index."""
        return os.getenv("PINECONE_WORKFLOW_INDEX")

    @property
    def bucket_name(self):
        """Return the bucket name."""
        return os.getenv("BUCKET_NAME")

    @property
    def access_key(self):
        """Return the access key."""
        return os.getenv("ACCESS_KEY")

    @property
    def secret_access_key(self):
        """Return the secret access key."""
        return os.getenv("SECRET_ACCESS_KEY")

    @property
    def default_file_storage(self):
        """Return the default file storage."""
        return os.getenv("DEFAULT_FILE_STORAGE")


# Create an instance of EnvLoader
env_loader = EnvLoader()
