"""
Data ingestion pipeline for automotive supply chain AI system.

This module provides a complete pipeline to load, enrich, and upload
automotive parts data to Qdrant Cloud with multitenancy support.
"""

import time
import hashlib
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from loguru import logger

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data_loader import load_all_data, enrich_parts_data
from src.embeddings import EmbeddingGenerator
from src.qdrant_client import (
    get_qdrant_client,
    recreate_collection,
    get_collection_stats,
)
from src.config import settings

# Import PointStruct after our modules to avoid circular import
# Use importlib to ensure we get the installed package, not our src/qdrant_client.py
import importlib
_qdrant_models = importlib.import_module('qdrant_client.models')
PointStruct = _qdrant_models.PointStruct


def ingest_data() -> Dict[str, Any]:
    """
    Complete pipeline to load data and upload to Qdrant Cloud with multitenancy.
    
    This function:
    1. Loads all data from JSON files
    2. Enriches parts data with suppliers and relationships
    3. Generates embeddings for parts
    4. Uploads to Qdrant with multitenancy support (Qdrant 1.16)
    5. Verifies data distribution across tenants
    
    Multitenancy: Each OEM's parts stored with oem_id for tenant isolation.
    Qdrant 1.16 handles routing automatically based on oem_id field.
    
    Returns:
        Dictionary containing summary statistics:
        - total_points: Total number of points uploaded
        - points_per_oem: Dictionary mapping OEM IDs to point counts
        - upload_time: Total time taken for upload (seconds)
        - average_embedding_time: Average time per embedding (seconds)
        - oems_processed: List of OEM IDs processed
        
    Raises:
        RuntimeError: If ingestion fails at any stage
        
    Example:
        >>> stats = ingest_data()
        >>> print(f"Uploaded {stats['total_points']} points")
    """
    start_time = time.time()
    
    try:
        logger.info("="*60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("="*60)
        
        # Step 1: Load all data
        logger.info("\n[Step 1/5] Loading data files...")
        data = load_all_data()
        
        parts = data.get('parts', [])
        suppliers = data.get('suppliers', [])
        relationships = data.get('relationships', [])
        oems = data.get('oems', [])
        
        if not parts:
            raise RuntimeError("No parts data loaded")
        
        logger.success(f"✓ Loaded {len(parts)} parts, {len(suppliers)} suppliers, "
                      f"{len(relationships)} relationships, {len(oems)} OEMs")
        
        # Step 2: Enrich parts data
        logger.info("\n[Step 2/5] Enriching parts data...")
        enriched_parts = enrich_parts_data(parts, suppliers, relationships)
        logger.success(f"✓ Enriched {len(enriched_parts)} parts")
        
        # Step 3: Create EmbeddingGenerator
        logger.info("\n[Step 3/5] Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator()
        metadata = embedding_generator.get_embedding_metadata()
        logger.success(f"✓ Embedding generator ready (model: {metadata['model_name']}, "
                      f"vector size: {metadata['vector_size']})")
        
        # Step 4: Recreate collection (fresh start with all 1.16 features)
        logger.info("\n[Step 4/5] Recreating Qdrant collection...")
        client = get_qdrant_client()
        recreate_result = recreate_collection(client)
        logger.success("✓ Collection recreated with Qdrant 1.16 features")
        
        # Step 5: Upload data with multitenancy
        logger.info("\n[Step 5/5] Uploading data to Qdrant with multitenancy...")
        logger.info("Multitenancy: Each OEM's parts stored with oem_id for tenant isolation")
        logger.info("Qdrant 1.16 handles routing automatically based on oem_id field")
        
        # Get unique OEM IDs from enriched parts
        oem_ids = sorted(set(part.get('oem_id') for part in enriched_parts if part.get('oem_id')))
        
        if not oem_ids:
            logger.warning("No OEM IDs found in parts, assigning default OEM")
            oem_ids = ['OEM-A']  # Fallback
        
        total_points_uploaded = 0
        points_per_oem: Dict[str, int] = {}
        embedding_start_time = time.time()
        total_embeddings_generated = 0
        
        # Process each OEM separately for multitenancy
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=None,  # Use default console
        ) as progress:
            
            for oem_id in oem_ids:
                # Filter parts for this OEM
                oem_parts = [
                    part for part in enriched_parts
                    if part.get('oem_id') == oem_id
                ]
                
                if not oem_parts:
                    logger.warning(f"No parts found for OEM {oem_id}, skipping...")
                    continue
                
                logger.info(f"\nProcessing OEM: {oem_id} ({len(oem_parts)} parts)")
                
                # Extract search_text from each part
                search_texts = [
                    part.get('search_text', '') or part.get('part_name', '')
                    for part in oem_parts
                ]
                
                # Generate embeddings in batch
                task_embed = progress.add_task(
                    f"[cyan]Generating embeddings for {oem_id}...",
                    total=len(search_texts)
                )
                
                embeddings = embedding_generator.generate_embeddings_batch(
                    search_texts,
                    batch_size=32
                )
                
                progress.update(task_embed, completed=len(search_texts))
                total_embeddings_generated += len(embeddings)
                
                logger.success(f"✓ Generated {len(embeddings)} embeddings for {oem_id}")
                
                # Prepare Qdrant points
                task_prep = progress.add_task(
                    f"[yellow]Preparing points for {oem_id}...",
                    total=len(oem_parts)
                )
                
                points = []
                for idx, (part, embedding) in enumerate(zip(oem_parts, embeddings)):
                    # Generate point ID: hash of part_id or use counter
                    part_id = part.get('part_id', f'part_{idx}')
                    point_id = int(hashlib.md5(part_id.encode()).hexdigest()[:8], 16)
                    
                    # Prepare payload with ALL part fields including oem_id (critical for multitenancy)
                    payload = part.copy()
                    
                    # Ensure oem_id is in payload (critical for multitenancy)
                    if 'oem_id' not in payload:
                        payload['oem_id'] = oem_id
                    
                    # Create PointStruct for Qdrant
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                    progress.update(task_prep, advance=1)
                
                progress.remove_task(task_prep)
                
                # Upload to Qdrant in batches of 100 points
                batch_size = 100
                total_batches = (len(points) + batch_size - 1) // batch_size
                
                task_upload = progress.add_task(
                    f"[green]Uploading {oem_id} to Qdrant...",
                    total=len(points)
                )
                
                oem_points_uploaded = 0
                for batch_idx in range(0, len(points), batch_size):
                    batch = points[batch_idx:batch_idx + batch_size]
                    
                    try:
                        # Upload batch
                        client.upsert(
                            collection_name=settings.collection_name,
                            points=batch
                        )
                        
                        oem_points_uploaded += len(batch)
                        progress.update(task_upload, advance=len(batch))
                        
                    except Exception as e:
                        logger.error(f"Failed to upload batch {batch_idx // batch_size + 1} "
                                   f"for {oem_id}: {e}")
                        raise
                
                progress.remove_task(task_upload)
                
                total_points_uploaded += oem_points_uploaded
                points_per_oem[oem_id] = oem_points_uploaded
                
                logger.success(f"✓ Uploaded {oem_points_uploaded} points for {oem_id}")
        
        embedding_time = time.time() - embedding_start_time
        total_time = time.time() - start_time
        
        # Verify distribution using get_collection_stats
        logger.info("\n" + "="*60)
        logger.info("Verifying data distribution...")
        logger.info("="*60)
        
        try:
            stats = get_collection_stats(client)
            
            logger.info(f"Total points in collection: {stats['total_points']}")
            logger.info(f"Total vectors: {stats['total_vectors']}")
            logger.info(f"Tenants (OEMs): {stats['tenants']['count']}")
            
            if stats['tenants']['distribution']:
                logger.info("\nPoints per tenant (OEM):")
                for oem_id, count in sorted(stats['tenants']['distribution'].items()):
                    logger.info(f"  {oem_id}: {count} points")
            
            logger.info(f"\nShards: {stats['shards']['count']} "
                       f"(Qdrant 1.16 Tiered Multitenancy)")
            
        except Exception as e:
            logger.warning(f"Could not retrieve collection stats: {e}")
        
        # Calculate average embedding time
        average_embedding_time = (
            embedding_time / total_embeddings_generated
            if total_embeddings_generated > 0 else 0
        )
        
        # Summary statistics
        summary = {
            'total_points': total_points_uploaded,
            'points_per_oem': points_per_oem,
            'upload_time': total_time,
            'average_embedding_time': average_embedding_time,
            'oems_processed': oem_ids,
            'total_embeddings': total_embeddings_generated,
        }
        
        logger.info("\n" + "="*60)
        logger.info("Data Ingestion Complete!")
        logger.info("="*60)
        logger.info(f"Total points uploaded: {total_points_uploaded}")
        logger.info(f"Points per OEM:")
        for oem_id, count in sorted(points_per_oem.items()):
            logger.info(f"  {oem_id}: {count} points")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average embedding time: {average_embedding_time*1000:.2f} ms per embedding")
        logger.info("="*60)
        
        return summary
        
    except Exception as e:
        error_msg = f"Data ingestion failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def main():
    """Main entry point for data ingestion."""
    try:
        stats = ingest_data()
        print("\n✅ Data ingestion completed successfully!")
        print(f"   Uploaded {stats['total_points']} points across {len(stats['oems_processed'])} OEMs")
        return 0
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

