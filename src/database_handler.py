from chromadb import Client, PersistentClient


class DatabaseHandler:
    def __init__(self, is_persistent: bool = False, database_path: str = "databases/"):
        self.is_persistent = is_persistent
        self.database_path = database_path
        self.chroma_client = self._create_client()

    def _create_client(self):
        if not self.is_persistent:
            return Client()
        return PersistentClient(path=self.database_path)
