import os
from dotenv import load_dotenv


class ConfigHandler:
    def __init__(self):
        load_dotenv()
        # self.something = _get_env(something)

    def _get_env(self, env: str):
        secret = os.getenv(env)
        if secret is None:
            raise Exception(f"secret: '{env}' does not exist")
        return secret
