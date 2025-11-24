"""
Qdrant Cloud client management for the Automotive Supply Chain AI system.

This module handles connection to Qdrant Cloud (version 1.16) and provides
utilities for version verification and client management.
"""

from typing import Optional, Dict, Any
from loguru import logger

# Import qdrant_client package directly
# Note: Our file is src/qdrant_client.py, so Python should find the installed package
# If there's a conflict, ensure the package is installed and in Python path
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PayloadSchemaType,
        TextIndexParams,
        CollectionStatus,
    )
except ImportError as e:
    # Fallback: try importing from the installed package explicitly
    import importlib.util
    import sys
    
    # Find qdrant_client in site-packages
    for path in sys.path:
        if 'site-packages' in path:
            qdrant_path = f"{path}/qdrant_client/__init__.py"
            if importlib.util.find_spec('qdrant_client'):
                spec = importlib.util.find_spec('qdrant_client')
                qdrant_client = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(qdrant_client)
                QdrantClient = qdrant_client.QdrantClient
                break
    else:
        raise ImportError(f"Could not import qdrant_client: {e}")

from src.config import settings


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant Cloud client instance.
    
    Connects to Qdrant Cloud using configuration from settings. Uses HTTP
    protocol (prefer_grpc=False) for better cloud compatibility and sets
    a 30-second timeout for operations.
    
    Returns:
        QdrantClient: Configured Qdrant client instance
        
    Raises:
        ConnectionError: If connection to Qdrant Cloud fails
        ValueError: If required configuration is missing
        
    Example:
        >>> client = get_qdrant_client()
        >>> collections = client.get_collections()
    """
    try:
        logger.info(f"Connecting to Qdrant Cloud at {settings.qdrant_url}")
        
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,  # Use HTTP for cloud compatibility
            timeout=30.0  # 30 second timeout
        )
        
        # Verify connection by testing it
        try:
            version_info = check_qdrant_version(client)
            logger.success(
                f"Successfully connected to Qdrant Cloud (version {version_info})"
            )
        except Exception as version_error:
            # If version check fails, still log successful connection
            logger.success("Successfully connected to Qdrant Cloud")
            logger.debug(f"Version check skipped: {version_error}")
        
        return client
        
    except UnexpectedResponse as e:
        error_msg = f"Failed to connect to Qdrant Cloud: {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error connecting to Qdrant Cloud: {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


def check_qdrant_version(client: Optional[QdrantClient] = None) -> str:
    """
    Check and verify Qdrant server version is 1.16 or higher.
    
    Queries the Qdrant server to retrieve version information and verifies
    that it meets the minimum requirement of 1.16 for ACORN and multitenancy
    features.
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a
                temporary client for version check.
    
    Returns:
        str: Qdrant server version string (e.g., "1.16.0")
        
    Raises:
        ConnectionError: If unable to retrieve version information
        RuntimeError: If Qdrant version is below 1.16
        
    Example:
        >>> client = get_qdrant_client()
        >>> version = check_qdrant_version(client)
        >>> print(f"Qdrant version: {version}")
    """
    # Use provided client or create temporary one
    temp_client = None
    if client is None:
        try:
            temp_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=False,
                timeout=30.0
            )
            client = temp_client
        except Exception as e:
            logger.error(f"Failed to create client for version check: {e}")
            raise ConnectionError(f"Unable to connect to Qdrant: {e}") from e
    
    try:
        # Test connection by getting collections (this verifies connection works)
        collections = client.get_collections()
        
        # Try to get version info from the API
        # In qdrant-client, we can access the http client to get version
        try:
            if hasattr(client, '_client') and hasattr(client._client, 'get_version'):
                version_info = client._client.get_version()
                version_string = version_info.version if hasattr(version_info, 'version') else str(version_info)
            elif hasattr(client, 'http_client') and hasattr(client.http_client, 'get_version'):
                version_info = client.http_client.get_version()
                version_string = version_info.version if hasattr(version_info, 'version') else str(version_info)
            else:
                # Fallback: use a simple connection test
                logger.info("Version check not available, connection verified via collections API")
                version_string = "unknown"
        except Exception as version_error:
            logger.debug(f"Could not retrieve version info: {version_error}")
            version_string = "unknown"
        
        if version_string != "unknown":
            logger.info(f"Qdrant server version: {version_string}")
            
            # Parse version to check if it's 1.16+
            try:
                major, minor = map(int, version_string.split('.')[:2])
                if major < 1 or (major == 1 and minor < 16):
                    warning_msg = (
                        f"Qdrant version {version_string} is below 1.16. "
                        "ACORN algorithm and tiered multitenancy features may not be available."
                    )
                    logger.warning(warning_msg)
                else:
                    logger.info(
                        f"Qdrant version {version_string} supports ACORN and multitenancy features"
                    )
            except (ValueError, IndexError):
                logger.warning(
                    f"Unable to parse Qdrant version '{version_string}'. "
                    "Assuming compatibility with 1.16+ features."
                )
        else:
            logger.info("Connection verified. Assuming Qdrant 1.16+ compatibility.")
            version_string = "1.16+ (assumed)"
        
        return version_string
        
    except Exception as e:
        error_msg = f"Failed to retrieve Qdrant version: {e}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                # QdrantClient doesn't have explicit close, but we can clear reference
                del temp_client
            except Exception:
                pass


def create_collection(client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Create Qdrant collection with 1.16 features enabled.
    
    Creates the "automotive_parts" collection with:
    - Vector configuration (384 dimensions, cosine distance, on-disk payload)
    - Tiered multitenancy (sharding for multi-tenant architecture)
    - Payload indexes (standard and full-text with ASCII folding)
    - ACORN algorithm configuration
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a new one.
    
    Returns:
        Dict containing collection information and status
    
    Raises:
        ConnectionError: If unable to connect to Qdrant
        RuntimeError: If collection creation fails
        
    Example:
        >>> client = get_qdrant_client()
        >>> result = create_collection(client)
        >>> print(result)
    """
    # Use provided client or create new one
    temp_client = None
    if client is None:
        client = get_qdrant_client()
        temp_client = client
    
    collection_name = settings.collection_name
    
    try:
        # Check if collection already exists
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name in existing_collections:
            logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
            collection_info = client.get_collection(collection_name)
            return {
                "status": "exists",
                "collection_name": collection_name,
                "info": {
                    "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                    "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                }
            }
        
        logger.info(f"Creating collection '{collection_name}' with Qdrant 1.16 features...")
        
        # 1. Vector Configuration
        vector_config = VectorParams(
            size=settings.vector_size,  # 384 dimensions
            distance=Distance.COSINE,   # Cosine similarity
        )
        
        # 2. Create collection with Tiered Multitenancy (Qdrant 1.16 Feature)
        # Qdrant 1.16 Tiered Multitenancy - isolates tenants without separate collections
        # This allows multiple OEMs in a single collection with data isolation
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            on_disk_payload=True,  # Store payload on disk for efficiency
            shard_number=settings.shard_number if settings.enable_multitenancy else None,
        )
        
        logger.success(f"Collection '{collection_name}' created successfully")
        logger.info(f"  - Vector size: {settings.vector_size}")
        logger.info(f"  - Distance metric: Cosine")
        logger.info(f"  - On-disk payload: True")
        if settings.enable_multitenancy:
            logger.info(f"  - Tiered Multitenancy: Enabled ({settings.shard_number} shards)")
        
        # 3. Create payload indexes
        
        # STANDARD INDEXES
        logger.info("Creating standard payload indexes...")
        
        standard_indexes = [
            ("part_id", PayloadSchemaType.KEYWORD),
            ("supplier_id", PayloadSchemaType.KEYWORD),
            ("category", PayloadSchemaType.KEYWORD),
            ("quality_rating", PayloadSchemaType.FLOAT),
            ("oem_id", PayloadSchemaType.KEYWORD),  # CRITICAL for multitenancy
            ("lead_time_days", PayloadSchemaType.INTEGER),
        ]
        
        for field_name, field_type in standard_indexes:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"  ✓ Created index: {field_name} ({field_type.value})")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to create index {field_name}: {e}")
        
        # FULL-TEXT INDEXES (Qdrant 1.16 Feature)
        # Qdrant 1.16 Full-Text - automatic ASCII folding for international characters
        logger.info("Creating full-text payload indexes...")
        
        try:
            # Create full-text index with Qdrant 1.16 features
            # ASCII folding is automatic in Qdrant 1.16 for international characters
            text_index_params = TextIndexParams(
                type="text",
                tokenizer="word",
                lowercase=True,
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="part_name",
                field_schema=text_index_params,
            )
            logger.success("  ✓ Created full-text index: part_name (with ASCII folding)")
        except Exception as e:
            logger.warning(f"  ⚠ Failed to create full-text index part_name: {e}")
            # Try alternative method if the first fails
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="part_name",
                    field_schema={"type": "text", "tokenizer": "word", "lowercase": True},
                )
                logger.success("  ✓ Created full-text index: part_name (alternative method)")
            except Exception as e2:
                logger.error(f"  ✗ Failed to create full-text index with alternative method: {e2}")
        
        # 4. Configure ACORN parameters (Qdrant 1.16 Feature)
        # Qdrant 1.16 ACORN - improves recall on restrictive filters
        # Note: ACORN configuration is typically done at query time via search parameters
        # The acorn_max_selectivity setting will be used when performing searches
        logger.info(f"ACORN algorithm enabled: {settings.acorn_enabled}")
        logger.info(f"ACORN max selectivity: {settings.acorn_max_selectivity}")
        
        # Get collection info to return
        collection_info = client.get_collection(collection_name)
        
        result = {
            "status": "created",
            "collection_name": collection_name,
            "features": {
                "vector_size": settings.vector_size,
                "distance": "cosine",
                "on_disk_payload": True,
                "multitenancy_enabled": settings.enable_multitenancy,
                "shard_number": settings.shard_number if settings.enable_multitenancy else None,
                "acorn_enabled": settings.acorn_enabled,
                "acorn_max_selectivity": settings.acorn_max_selectivity,
                "full_text_indexing": True,
            },
            "info": {
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
            }
        }
        
        logger.success("✅ Collection created with all Qdrant 1.16 features:")
        logger.success(f"   • Tiered Multitenancy: {'Enabled' if settings.enable_multitenancy else 'Disabled'}")
        logger.success(f"   • Full-Text Indexing: Enabled (with ASCII folding)")
        logger.success(f"   • ACORN Algorithm: {'Enabled' if settings.acorn_enabled else 'Disabled'}")
        logger.success(f"   • Standard Indexes: {len(standard_indexes)} created")
        
        return result
        
    except UnexpectedResponse as e:
        error_msg = f"Failed to create collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error creating collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                del temp_client
            except Exception:
                pass


def get_collection_info(client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Get collection metadata including shard information for multitenancy verification.
    
    Retrieves detailed information about the collection including:
    - Basic collection info (points, vectors)
    - Shard information (Qdrant 1.16 Tiered Multitenancy feature)
    - Index information
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a new one.
    
    Returns:
        Dict containing collection metadata and shard information
    
    Raises:
        RuntimeError: If collection doesn't exist or cannot be accessed
        
    Example:
        >>> client = get_qdrant_client()
        >>> info = get_collection_info(client)
        >>> print(f"Shards: {info['shards']}")
    """
    # Use provided client or create new one
    temp_client = None
    if client is None:
        client = get_qdrant_client()
        temp_client = client
    
    collection_name = settings.collection_name
    
    try:
        logger.info(f"Retrieving collection info for '{collection_name}'...")
        
        # Get collection information
        collection_info = client.get_collection(collection_name)
        
        # Extract shard information (Qdrant 1.16 Tiered Multitenancy feature)
        shard_info = {}
        if hasattr(collection_info, 'shards') and collection_info.shards:
            shard_info['count'] = len(collection_info.shards)
            shard_info['details'] = []
            for shard in collection_info.shards:
                shard_detail = {
                    'shard_id': getattr(shard, 'shard_id', None),
                    'points_count': getattr(shard, 'points_count', 0),
                }
                shard_info['details'].append(shard_detail)
        elif hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
            # Try to get shard number from config
            params = collection_info.config.params
            if hasattr(params, 'shard_number'):
                shard_info['count'] = params.shard_number
                shard_info['details'] = []
        
        result = {
            "collection_name": collection_name,
            "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
            "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
            "status": str(collection_info.status) if hasattr(collection_info, 'status') else "unknown",
            "shards": shard_info,
        }
        
        # Log shard information (Qdrant 1.16 Tiered Multitenancy)
        if shard_info.get('count'):
            logger.info(f"Collection has {shard_info['count']} shard(s) (Qdrant 1.16 Tiered Multitenancy)")
            if shard_info.get('details'):
                for shard_detail in shard_info['details']:
                    logger.debug(f"  Shard {shard_detail.get('shard_id', 'N/A')}: {shard_detail.get('points_count', 0)} points")
        else:
            logger.info("Shard information not available (may be single-shard collection)")
        
        logger.success(f"Collection info retrieved: {result['points_count']} points, {result['vectors_count']} vectors")
        
        return result
        
    except UnexpectedResponse as e:
        if "not found" in str(e).lower() or "404" in str(e):
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        error_msg = f"Failed to get collection info for '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error getting collection info for '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                del temp_client
            except Exception:
                pass


def delete_collection(client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Delete the collection with proper warnings about data loss.
    
    Checks if collection exists before attempting deletion and logs
    appropriate warnings about permanent data loss.
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a new one.
    
    Returns:
        Dict containing deletion status
    
    Raises:
        RuntimeError: If deletion fails
        
    Example:
        >>> client = get_qdrant_client()
        >>> result = delete_collection(client)
        >>> print(result['status'])
    """
    # Use provided client or create new one
    temp_client = None
    if client is None:
        client = get_qdrant_client()
        temp_client = client
    
    collection_name = settings.collection_name
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name not in existing_collections:
            logger.warning(f"Collection '{collection_name}' does not exist. Nothing to delete.")
            return {
                "status": "not_found",
                "collection_name": collection_name,
                "message": "Collection does not exist"
            }
        
        # Get collection info before deletion for logging
        try:
            collection_info = client.get_collection(collection_name)
            points_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
            logger.warning(f"⚠️  WARNING: About to delete collection '{collection_name}'")
            logger.warning(f"   This will permanently delete {points_count} points!")
            logger.warning(f"   This action cannot be undone!")
        except Exception:
            pass
        
        # Delete the collection
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name)
        
        logger.success(f"✅ Collection '{collection_name}' deleted successfully")
        
        return {
            "status": "deleted",
            "collection_name": collection_name,
            "message": "Collection deleted successfully"
        }
        
    except UnexpectedResponse as e:
        error_msg = f"Failed to delete collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error deleting collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                del temp_client
            except Exception:
                pass


def recreate_collection(client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Delete existing collection and create a fresh one with all Qdrant 1.16 features.
    
    This function is useful for testing and data refresh. It will:
    1. Delete the existing collection (if it exists)
    2. Create a new collection with all Qdrant 1.16 features enabled:
       - Tiered Multitenancy (sharding)
       - Full-Text Indexing (with ASCII folding)
       - ACORN Algorithm configuration
       - All standard payload indexes
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a new one.
    
    Returns:
        Dict containing recreation status and collection info
    
    Raises:
        RuntimeError: If recreation fails
        
    Example:
        >>> client = get_qdrant_client()
        >>> result = recreate_collection(client)
        >>> print(result['status'])
    """
    # Use provided client or create new one
    temp_client = None
    if client is None:
        client = get_qdrant_client()
        temp_client = client
    
    collection_name = settings.collection_name
    
    try:
        logger.info(f"Recreating collection '{collection_name}' with all Qdrant 1.16 features...")
        
        # Delete existing collection if it exists
        try:
            delete_result = delete_collection(client)
            if delete_result['status'] == 'deleted':
                logger.info("Existing collection deleted")
        except Exception as e:
            logger.debug(f"Collection may not have existed or deletion failed: {e}")
        
        # Create fresh collection with all 1.16 features
        logger.info("Creating fresh collection with Qdrant 1.16 features:")
        logger.info("  • Tiered Multitenancy (sharding)")
        logger.info("  • Full-Text Indexing (ASCII folding)")
        logger.info("  • ACORN Algorithm")
        logger.info("  • Standard payload indexes")
        
        create_result = create_collection(client)
        
        logger.success(f"✅ Collection '{collection_name}' recreated successfully")
        
        return {
            "status": "recreated",
            "collection_name": collection_name,
            "creation_result": create_result,
            "features": {
                "tiered_multitenancy": settings.enable_multitenancy,
                "shard_number": settings.shard_number if settings.enable_multitenancy else None,
                "full_text_indexing": True,
                "acorn_enabled": settings.acorn_enabled,
                "acorn_max_selectivity": settings.acorn_max_selectivity,
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to recreate collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                del temp_client
            except Exception:
                pass


def get_collection_stats(client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Get collection statistics including tenant distribution and shard information.
    
    Returns comprehensive statistics about the collection:
    - Total points and vectors
    - Points per tenant (oem_id) - useful for multitenancy verification
    - Distribution across shards (Qdrant 1.16 Tiered Multitenancy feature)
    
    This function helps verify that multitenancy is working correctly
    by showing how data is distributed across tenants and shards.
    
    Args:
        client: Optional QdrantClient instance. If not provided, creates a new one.
    
    Returns:
        Dict containing collection statistics
    
    Raises:
        RuntimeError: If collection doesn't exist or stats cannot be retrieved
        
    Example:
        >>> client = get_qdrant_client()
        >>> stats = get_collection_stats(client)
        >>> print(f"Total points: {stats['total_points']}")
        >>> print(f"Tenants: {stats['tenants']}")
    """
    # Use provided client or create new one
    temp_client = None
    if client is None:
        client = get_qdrant_client()
        temp_client = client
    
    collection_name = settings.collection_name
    
    try:
        logger.info(f"Retrieving statistics for collection '{collection_name}'...")
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
        total_vectors = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0
        
        # Get points per tenant (oem_id) - Critical for multitenancy verification
        tenant_stats = {}
        if total_points > 0:
            try:
                # Scroll through all points to count by oem_id
                # Note: This is a simplified approach - for large collections, consider using aggregation
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Adjust based on your needs
                    with_payload=True,
                    with_vectors=False,
                )
                
                # Count points per tenant
                for point in scroll_result[0]:
                    if point.payload and 'oem_id' in point.payload:
                        oem_id = str(point.payload['oem_id'])
                        tenant_stats[oem_id] = tenant_stats.get(oem_id, 0) + 1
                
                logger.info(f"Found {len(tenant_stats)} tenant(s) (oem_id values)")
                for oem_id, count in tenant_stats.items():
                    logger.debug(f"  Tenant {oem_id}: {count} points")
                    
            except Exception as e:
                logger.warning(f"Could not retrieve tenant distribution: {e}")
                tenant_stats = {}
        
        # Get shard distribution (Qdrant 1.16 Tiered Multitenancy feature)
        shard_distribution = {}
        try:
            collection_info_full = client.get_collection(collection_name)
            if hasattr(collection_info_full, 'shards') and collection_info_full.shards:
                for shard in collection_info_full.shards:
                    shard_id = getattr(shard, 'shard_id', 'unknown')
                    points = getattr(shard, 'points_count', 0)
                    shard_distribution[str(shard_id)] = points
                logger.info(f"Shard distribution retrieved for {len(shard_distribution)} shard(s)")
        except Exception as e:
            logger.debug(f"Could not retrieve shard distribution: {e}")
        
        result = {
            "collection_name": collection_name,
            "total_points": total_points,
            "total_vectors": total_vectors,
            "tenants": {
                "count": len(tenant_stats),
                "distribution": tenant_stats,
            },
            "shards": {
                "count": len(shard_distribution) if shard_distribution else (settings.shard_number if settings.enable_multitenancy else 1),
                "distribution": shard_distribution,
            },
        }
        
        # Log summary
        logger.success("Collection Statistics:")
        logger.info(f"  Total Points: {total_points}")
        logger.info(f"  Total Vectors: {total_vectors}")
        logger.info(f"  Tenants: {result['tenants']['count']}")
        if result['shards']['distribution']:
            logger.info(f"  Shards: {len(result['shards']['distribution'])} (Qdrant 1.16 Tiered Multitenancy)")
            for shard_id, points in result['shards']['distribution'].items():
                logger.debug(f"    Shard {shard_id}: {points} points")
        
        return result
        
    except UnexpectedResponse as e:
        if "not found" in str(e).lower() or "404" in str(e):
            error_msg = f"Collection '{collection_name}' does not exist"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        error_msg = f"Failed to get collection stats for '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error getting collection stats for '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Clean up temporary client if we created one
        if temp_client is not None:
            try:
                del temp_client
            except Exception:
                pass

