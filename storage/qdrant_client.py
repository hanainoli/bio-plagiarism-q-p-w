"""
Qdrant Storage Client.
Handles all vector database operations for plagiarism detection:
- Collection management
- Point insertion and retrieval
- Vector search
- Batch operations
"""

import os
import time
import random
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 10.0  # seconds
RETRYABLE_STATUS_CODES = {502, 503, 504, 429}  # Bad Gateway, Service Unavailable, Gateway Timeout, Too Many Requests


def retry_with_backoff(func, max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY):
    """
    Retry a function with exponential backoff for transient errors.
    
    Args:
        func: Function to call (should be a lambda or callable)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be exponentially increased)
    
    Returns:
        Result of the function call
    
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            # Check if it's a retryable error
            is_retryable = False
            for code in RETRYABLE_STATUS_CODES:
                if str(code) in error_str or 'Bad Gateway' in error_str or 'Service Unavailable' in error_str:
                    is_retryable = True
                    break
            
            # Also retry on connection errors
            if 'Connection' in error_str or 'Timeout' in error_str or 'timeout' in error_str:
                is_retryable = True
            
            if is_retryable and attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), RETRY_MAX_DELAY)
                print(f"  ⚠ Transient error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:100]}")
                print(f"    Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Non-retryable error or max retries reached
                raise
    
    raise last_exception


@dataclass
class SearchResult:
    """Search result from Qdrant"""
    id: str
    score: float
    payload: Dict[str, Any]


class QdrantClient:
    """Client for Qdrant vector database operations"""
    
    # Collection configurations
    COLLECTIONS = {
        'abstracts': {
            'vector_size': 1024,
            'distance': 'Cosine'
        },
        'fulltext_chunks': {
            'vector_size': 1024,
            'distance': 'Cosine'
        },
        'figures': {
            'vectors_config': {
                'cnn_vector': {'size': 2048, 'distance': 'Cosine'},
                'clip_vector': {'size': 512, 'distance': 'Cosine'}
            }
        },
        'sub_figures': {
            'vector_size': 2048,
            'distance': 'Cosine'
        },
        'tables': {
            'vector_size': 2048,  # ResNet image embeddings
            'distance': 'Cosine'
        }
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        https: Optional[bool] = None
    ):
        """
        Initialize Qdrant client.
        Reads from environment variables if not provided.
        
        Args:
            host: Qdrant server host (or QDRANT_HOST env var)
            port: Qdrant server port (or QDRANT_PORT env var)
            api_key: API key for authentication (or QDRANT_API_KEY env var)
            https: Use HTTPS connection (or QDRANT_HTTPS env var)
        """
        # Read from environment with fallback to defaults
        self.host = host or os.environ.get('QDRANT_HOST', 'localhost')
        self.port = port or int(os.environ.get('QDRANT_PORT', 6333))
        self.api_key = api_key or os.environ.get('QDRANT_API_KEY')
        
        # Handle HTTPS setting
        if https is not None:
            self.https = https
        else:
            https_env = os.environ.get('QDRANT_HTTPS', 'false').lower()
            self.https = https_env in ('true', '1', 'yes')
        
        self._client = None
    
    @property
    def client(self):
        """Lazy load Qdrant client"""
        if self._client is None:
            self._client = self._connect()
        return self._client
    
    def _connect(self):
        """Connect to Qdrant server"""
        try:
            from qdrant_client import QdrantClient as QC
            from qdrant_client.http import models
            
            # Check if host is a URL (Qdrant Cloud)
            if self.host.startswith('http://') or self.host.startswith('https://'):
                # Cloud connection with URL
                print(f"Connecting to Qdrant Cloud: {self.host[:50]}...")
                return QC(
                    url=self.host,
                    api_key=self.api_key
                )
            elif self.api_key:
                # Cloud connection with host + api_key
                url = f"https://{self.host}" if self.https else f"http://{self.host}"
                if self.port and self.port != 6333:
                    url = f"{url}:{self.port}"
                print(f"Connecting to Qdrant Cloud: {url[:50]}...")
                return QC(
                    url=url,
                    api_key=self.api_key
                )
            else:
                # Local connection
                print(f"Connecting to local Qdrant: {self.host}:{self.port}")
                return QC(
                    host=self.host,
                    port=self.port,
                    https=self.https
                )
        except ImportError:
            print("Warning: qdrant-client not installed")
            return None
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            return None
    
    # =========================================================================
    # COLLECTION MANAGEMENT
    # =========================================================================
    
    def create_collection(self, name: str, config: Optional[Dict] = None) -> bool:
        """
        Create a collection with specified configuration.
        
        Args:
            name: Collection name
            config: Collection configuration (uses default if None)
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            from qdrant_client.http import models
            
            if config is None:
                config = self.COLLECTIONS.get(name, {})
            
            # Check if multi-vector collection
            if 'vectors_config' in config:
                vectors_config = {}
                for vec_name, vec_config in config['vectors_config'].items():
                    vectors_config[vec_name] = models.VectorParams(
                        size=vec_config['size'],
                        distance=models.Distance.COSINE
                    )
                
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=vectors_config
                )
            else:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=config.get('vector_size', 1024),
                        distance=models.Distance.COSINE
                    )
                )
            
            return True
        
        except Exception as e:
            print(f"Error creating collection {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        if self.client is None:
            return False
        
        try:
            self.client.delete_collection(collection_name=name)
            return True
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists"""
        if self.client is None:
            return False
        
        try:
            collections = self.client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception:
            return False
    
    def get_collection_info(self, name: str) -> Optional[Dict]:
        """Get collection information"""
        if self.client is None:
            return None
        
        try:
            info = self.client.get_collection(collection_name=name)
            return {
                'name': name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception:
            return None
    
    def init_all_collections(self) -> bool:
        """Initialize all required collections"""
        success = True
        for name in self.COLLECTIONS.keys():
            if not self.collection_exists(name):
                if not self.create_collection(name):
                    success = False
        return success
    
    # =========================================================================
    # POINT OPERATIONS
    # =========================================================================
    
    def upsert_point(
        self,
        collection: str,
        point_id: str,
        vector: Union[np.ndarray, List[float], Dict[str, Any]],
        payload: Dict[str, Any]
    ) -> bool:
        """
        Insert or update a single point.
        
        Args:
            collection: Collection name
            point_id: Unique point ID (will be converted to UUID)
            vector: Vector data (single vector or dict for multi-vector)
            payload: Point payload/metadata
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            from qdrant_client.http import models
            import uuid
            
            # Convert string ID to UUID (deterministic based on string)
            if isinstance(point_id, str):
                uuid_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
            else:
                uuid_id = point_id
            
            # Store original ID in payload for reference
            payload['original_id'] = str(point_id)
            
            # Handle numpy arrays
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            elif isinstance(vector, dict):
                vector = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in vector.items()}
            
            # Ensure collection exists
            if not self.collection_exists(collection):
                self.create_collection(collection)
            
            self.client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=uuid_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        
        except Exception as e:
            print(f"Error upserting point: {e}")
            return False
    
    def upsert_batch(
        self,
        collection: str,
        points: List[Dict[str, Any]],
        wait: bool = True
    ) -> bool:
        """
        Batch insert/update multiple points efficiently.
        
        For best performance, this method sends points in chunks of 500
        to Qdrant in a single API call per chunk.
        
        Args:
            collection: Collection name
            points: List of dicts with 'id', 'vector', 'payload' keys
            wait: Wait for operation to complete
        
        Returns:
            True if successful
            
        Example:
            points = [
                {'id': 'point_1', 'vector': [0.1, 0.2, ...], 'payload': {'text': '...'}},
                {'id': 'point_2', 'vector': [0.3, 0.4, ...], 'payload': {'text': '...'}},
            ]
            client.upsert_batch('collection', points)
        """
        if self.client is None or not points:
            return False
        
        try:
            from qdrant_client.http import models
            import uuid
            
            # Ensure collection exists
            if not self.collection_exists(collection):
                self.create_collection(collection)
            
            # Convert to Qdrant PointStruct format
            qdrant_points = []
            for p in points:
                point_id = p.get('id', str(uuid.uuid4()))
                vector = p.get('vector')
                payload = p.get('payload', {})
                
                # Convert numpy arrays
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                elif isinstance(vector, dict):
                    vector = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in vector.items()}
                
                # Generate UUID from string ID
                if isinstance(point_id, str):
                    point_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, point_id)
                else:
                    point_uuid = point_id
                
                # Store original ID in payload
                payload['original_id'] = str(point_id)
                
                qdrant_points.append(models.PointStruct(
                    id=str(point_uuid),
                    vector=vector,
                    payload=payload
                ))
            
            # Upload in chunks of 500 (optimal for Qdrant) with retry
            chunk_size = 500
            for i in range(0, len(qdrant_points), chunk_size):
                chunk = qdrant_points[i:i + chunk_size]
                
                def do_upsert():
                    self.client.upsert(
                        collection_name=collection,
                        points=chunk,
                        wait=wait
                    )
                
                retry_with_backoff(do_upsert)
            
            return True
        
        except Exception as e:
            print(f"Error batch upserting to {collection} (after retries): {e}")
            return False
    
    def get_point(self, collection: str, point_id: str) -> Optional[Dict]:
        """
        Retrieve a single point by ID.
        
        Args:
            collection: Collection name
            point_id: Point ID
        
        Returns:
            Point data or None
        """
        if self.client is None:
            return None
        
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            
            if points:
                p = points[0]
                return {
                    'id': p.id,
                    'vector': p.vector,
                    'payload': p.payload
                }
            return None
        
        except Exception as e:
            print(f"Error retrieving point: {e}")
            return None
    
    def delete_point(self, collection: str, point_id: str) -> bool:
        """Delete a point by ID"""
        if self.client is None:
            return False
        
        try:
            from qdrant_client.http import models
            
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[point_id])
            )
            return True
        
        except Exception as e:
            print(f"Error deleting point: {e}")
            return False
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    def search(
        self,
        collection: str,
        vector: Union[np.ndarray, List[float]],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
        vector_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors with automatic retry for transient errors.
        
        Args:
            collection: Collection name
            vector: Query vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_conditions: Filter conditions for payload
            vector_name: Name of vector for multi-vector collections
        
        Returns:
            List of SearchResult objects
        """
        if self.client is None:
            return []
        
        try:
            from qdrant_client.http import models
            
            # Handle numpy arrays
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            # Build filter if conditions provided
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for field, value in filter_conditions.items():
                    if isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                query_filter = models.Filter(must=must_conditions)
            
            # Define search function for retry
            def do_search():
                # Try new API first (qdrant-client >= 1.7)
                try:
                    if vector_name:
                        return self.client.query_points(
                            collection_name=collection,
                            query=vector,
                            using=vector_name,
                            limit=limit,
                            score_threshold=score_threshold,
                            query_filter=query_filter,
                            with_payload=True
                        ).points
                    else:
                        return self.client.query_points(
                            collection_name=collection,
                            query=vector,
                            limit=limit,
                            score_threshold=score_threshold,
                            query_filter=query_filter,
                            with_payload=True
                        ).points
                except AttributeError:
                    # Fall back to old API (qdrant-client < 1.7)
                    if vector_name:
                        return self.client.search(
                            collection_name=collection,
                            query_vector=(vector_name, vector),
                            limit=limit,
                            score_threshold=score_threshold,
                            query_filter=query_filter,
                            with_payload=True
                        )
                    else:
                        return self.client.search(
                            collection_name=collection,
                            query_vector=vector,
                            limit=limit,
                            score_threshold=score_threshold,
                            query_filter=query_filter,
                            with_payload=True
                        )
            
            # Execute search with retry
            results = retry_with_backoff(do_search)
            
            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {}
                )
                for r in results
            ]
        
        except Exception as e:
            print(f"Error searching (after retries): {e}")
            return []
    
    def search_by_paper(
        self,
        collection: str,
        vector: Union[np.ndarray, List[float]],
        paper_id: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search within a specific paper"""
        return self.search(
            collection=collection,
            vector=vector,
            limit=limit,
            filter_conditions={'paper_id': paper_id}
        )
    
    def search_by_server(
        self,
        collection: str,
        vector: Union[np.ndarray, List[float]],
        server: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search within a specific server (biorxiv/medrxiv)"""
        return self.search(
            collection=collection,
            vector=vector,
            limit=limit,
            filter_conditions={'server': server}
        )
    
    # =========================================================================
    # SPECIALIZED OPERATIONS
    # =========================================================================
    
    def search_abstracts(
        self,
        vector: Union[np.ndarray, List[float]],
        limit: int = 20,
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search abstract collection"""
        return self.search(
            collection='bio_abstracts',
            vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
    
    def search_chunks(
        self,
        vector: Union[np.ndarray, List[float]],
        limit: int = 50,
        score_threshold: float = 0.5  # Lowered to catch more matches
    ) -> List[SearchResult]:
        """Search fulltext chunks collection"""
        return self.search(
            collection='bio_fulltext_chunks',
            vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
    
    def search_figures(
        self,
        cnn_vector: Union[np.ndarray, List[float]],
        limit: int = 20,
        score_threshold: float = 0.5  # Lowered from 0.8 to catch more matches
    ) -> List[SearchResult]:
        """Search figures by CNN vector"""
        return self.search(
            collection='bio_figures',
            vector=cnn_vector,
            limit=limit,
            score_threshold=score_threshold
        )
    
    def search_tables(
        self,
        vector: Union[np.ndarray, List[float]],
        limit: int = 20,
        score_threshold: float = 0.5  # Lowered from 0.8 to catch more matches
    ) -> List[SearchResult]:
        """Search tables collection"""
        return self.search(
            collection='bio_tables',
            vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def count_points(self, collection: str) -> int:
        """Get total points in collection"""
        info = self.get_collection_info(collection)
        return info.get('points_count', 0) if info else 0
    
    def scroll_all(
        self,
        collection: str,
        batch_size: int = 100,
        with_vectors: bool = False
    ):
        """
        Scroll through all points in collection.
        
        Args:
            collection: Collection name
            batch_size: Points per batch
            with_vectors: Include vectors in results
        
        Yields:
            Point data dictionaries
        """
        if self.client is None:
            return
        
        try:
            offset = None
            
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=with_vectors,
                    with_payload=True
                )
                
                for point in results:
                    yield {
                        'id': point.id,
                        'vector': point.vector if with_vectors else None,
                        'payload': point.payload
                    }
                
                if offset is None:
                    break
        
        except Exception as e:
            print(f"Error scrolling: {e}")


class MockQdrantClient(QdrantClient):
    """Mock client for testing without Qdrant server"""
    
    def __init__(self):
        self.collections = {}
        self._client = self
    
    def _connect(self):
        return self
    
    def create_collection(self, name: str, config: Optional[Dict] = None) -> bool:
        self.collections[name] = {'points': {}, 'config': config or {}}
        return True
    
    def delete_collection(self, name: str) -> bool:
        if name in self.collections:
            del self.collections[name]
        return True
    
    def collection_exists(self, name: str) -> bool:
        return name in self.collections
    
    def upsert_point(
        self,
        collection: str,
        point_id: str,
        vector: Union[np.ndarray, List[float], Dict],
        payload: Dict[str, Any]
    ) -> bool:
        if collection not in self.collections:
            self.create_collection(collection)
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        self.collections[collection]['points'][point_id] = {
            'id': point_id,
            'vector': vector,
            'payload': payload
        }
        return True
    
    def search(
        self,
        collection: str,
        vector: Union[np.ndarray, List[float]],
        limit: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        if collection not in self.collections:
            return []
        
        # Simple mock search - returns random scores
        points = list(self.collections[collection]['points'].values())[:limit]
        
        return [
            SearchResult(
                id=p['id'],
                score=0.9,
                payload=p['payload']
            )
            for p in points
        ]


if __name__ == "__main__":
    # Test with mock client
    client = MockQdrantClient()
    
    # Create collection
    client.create_collection('test_collection')
    
    # Insert point
    client.upsert_point(
        collection='test_collection',
        point_id='test_1',
        vector=np.random.randn(1024),
        payload={'paper_id': 'paper_123', 'text': 'Test content'}
    )
    
    # Search
    results = client.search(
        collection='test_collection',
        vector=np.random.randn(1024),
        limit=5
    )
    
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  ID: {r.id}, Score: {r.score:.3f}")
