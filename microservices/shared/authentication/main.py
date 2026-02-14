"""Authentication module for shared functionality."""

from jose import jwt

from envs.env_loader import EnvLoader

env_loader = EnvLoader()


async def verify_token(token: str):
    """Verify a JWT token and return the user ID."""
    decoded_token = jwt.decode(token, env_loader.secret_key, algorithms=["HS256"])
    user_id = decoded_token.get("unique_id")
    print("decoded_token", decoded_token)

    return user_id
