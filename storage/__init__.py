"""Storage modules for database and file storage."""

# Qdrant client (requires: qdrant-client)
try:
    from .qdrant_client import QdrantClient, MockQdrantClient, SearchResult
except ImportError as e:
    QdrantClient = None
    MockQdrantClient = None
    SearchResult = None
    print(f"Warning: Could not import qdrant_client (install qdrant-client): {e}")

# Wasabi client (requires: boto3)
try:
    from .wasabi_client import WasabiClient, MockWasabiClient
except ImportError as e:
    WasabiClient = None
    MockWasabiClient = None
    print(f"Warning: Could not import wasabi_client (install boto3): {e}")

__all__ = [
    'QdrantClient',
    'MockQdrantClient',
    'SearchResult',
    'WasabiClient',
    'MockWasabiClient',
]
