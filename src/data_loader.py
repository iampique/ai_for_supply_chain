"""
Data loader for automotive supply chain AI system.

This module handles loading JSON data files containing automotive parts,
suppliers, relationships, OEM profiles, and quality incidents.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load and parse a JSON file.
    
    Handles both list and dictionary JSON structures. If the JSON is a dictionary,
    it attempts to extract list data from common keys like 'data', 'items', or
    returns the dictionary wrapped in a list.
    
    Args:
        filepath: Path to the JSON file to load
    
    Returns:
        List of dictionaries containing the parsed data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the file structure is unexpected
        
    Example:
        >>> data = load_json_file('data/parts.json')
        >>> print(f"Loaded {len(data)} records")
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        error_msg = f"File not found: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.debug(f"Loading JSON file: {filepath}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of records
            result = data
        elif isinstance(data, dict):
            # Dictionary structure - try to extract list data
            # Check for common keys that contain lists
            list_keys = ['data', 'items', 'records', 'suppliers', 'oems', 
                        'quality_incidents', 'relationships', 'parts']
            
            result = None
            for key in list_keys:
                if key in data and isinstance(data[key], list):
                    result = data[key]
                    logger.debug(f"Extracted list from key '{key}'")
                    break
            
            # If no list found, wrap the dict in a list
            if result is None:
                logger.warning(f"No list found in dictionary structure, wrapping dict in list")
                result = [data]
        else:
            error_msg = f"Unexpected JSON structure: {type(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Loaded {len(result)} records from {filepath}")
        
        return result
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in file {filepath}: {e}"
        logger.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e
        
    except Exception as e:
        error_msg = f"Error loading file {filepath}: {e}"
        logger.error(error_msg)
        raise


def load_all_data(data_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all automotive supply chain data files.
    
    Loads the following JSON files from the data directory:
    - ev_hybrid_automotive_parts.json -> "parts"
    - suppliers.json -> "suppliers"
    - part_supplier_relationships.json -> "relationships"
    - oem_profiles.json -> "oems"
    - quality_incidents.json -> "incidents"
    
    Args:
        data_dir: Directory containing the JSON files. If None, automatically
                 detects the data directory relative to this file or project root.
    
    Returns:
        Dictionary with keys: "parts", "suppliers", "relationships", "oems", "incidents"
        Each value is a list of dictionaries containing the data records
    
    Example:
        >>> data = load_all_data()
        >>> print(f"Loaded {len(data['parts'])} parts")
        >>> print(f"Loaded {len(data['suppliers'])} suppliers")
    """
    # Auto-detect data directory if not provided
    if data_dir is None:
        # Try relative to this file first
        current_file = Path(__file__)
        data_path = current_file.parent.parent / "data"
        
        # If that doesn't exist, try current working directory
        if not data_path.exists():
            data_path = Path("data")
        
        # If still not found, try absolute path from common locations
        if not data_path.exists():
            # Try common project root locations
            for possible_root in [
                Path.cwd(),
                current_file.parent.parent.parent,
            ]:
                possible_data = possible_root / "data"
                if possible_data.exists():
                    data_path = possible_data
                    break
    else:
        data_path = Path(data_dir)
    
    if not data_path.exists():
        error_msg = f"Data directory not found: {data_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Define file mappings
    file_mappings = {
        "ev_hybrid_automotive_parts.json": "parts",
        "suppliers.json": "suppliers",
        "part_supplier_relationships.json": "relationships",
        "oem_profiles.json": "oems",
        "quality_incidents.json": "incidents",
    }
    
    result: Dict[str, List[Dict[str, Any]]] = {
        "parts": [],
        "suppliers": [],
        "relationships": [],
        "oems": [],
        "incidents": [],
    }
    
    logger.info("Loading automotive supply chain data files...")
    
    # Load each file
    for filename, key in file_mappings.items():
        filepath = data_path / filename
        
        try:
            if not filepath.exists():
                logger.warning(f"File not found: {filename}, skipping...")
                continue
            
            # Load the file (load_json_file handles nested dictionary extraction automatically)
            data = load_json_file(str(filepath))
            result[key] = data
            logger.success(f"✓ Loaded {len(data)} records from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            logger.warning(f"Continuing with other files...")
            # Keep empty list for this key
    
    # Log summary statistics
    logger.info("\n" + "="*60)
    logger.info("Data Loading Summary:")
    logger.info("="*60)
    total_records = 0
    for key, records in result.items():
        count = len(records)
        total_records += count
        logger.info(f"  {key.capitalize():15s}: {count:6d} records")
    logger.info("-"*60)
    logger.info(f"  {'Total':15s}: {total_records:6d} records")
    logger.info("="*60)
    
    return result


def enrich_parts_data(
    parts: List[Dict[str, Any]],
    suppliers: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine parts, suppliers, and relationships into enriched documents.
    
    For each part, this function:
    - Finds all relationships where part_id matches
    - Enriches with full supplier details
    - Creates a "search_text" field for full-text search
    - Ensures oem_id is present (for Qdrant 1.16 multitenancy)
    - Adds normalized_part_name for full-text search testing
    
    Args:
        parts: List of part dictionaries
        suppliers: List of supplier dictionaries
        relationships: List of part-supplier relationship dictionaries
    
    Returns:
        List of enriched part dictionaries with supplier information and search fields
    
    Example:
        >>> data = load_all_data()
        >>> enriched = enrich_parts_data(data['parts'], data['suppliers'], data['relationships'])
        >>> print(f"Enriched {len(enriched)} parts")
    """
    logger.info("Enriching parts data with suppliers and relationships...")
    
    # Convert suppliers list to dictionary keyed by supplier_id for O(1) lookups
    suppliers_dict = {supplier['supplier_id']: supplier for supplier in suppliers}
    logger.debug(f"Created suppliers lookup dictionary with {len(suppliers_dict)} suppliers")
    
    # Create relationships lookup by part_id using pandas for efficiency
    if relationships:
        relationships_df = pd.DataFrame(relationships)
        logger.debug(f"Created relationships DataFrame with {len(relationships_df)} relationships")
    else:
        relationships_df = pd.DataFrame()
        logger.warning("No relationships provided")
    
    # Get available OEM IDs for multitenancy assignment
    # Default OEM IDs: OEM-A, OEM-B, OEM-C (distribute evenly)
    oem_ids = ['OEM-A', 'OEM-B', 'OEM-C']
    
    enriched_parts = []
    total_parts = len(parts)
    
    logger.info(f"Processing {total_parts} parts...")
    
    for idx, part in enumerate(parts):
        # Create a copy to avoid modifying original
        enriched_part = part.copy()
        
        # Find all relationships for this part
        part_id = part.get('part_id', '')
        part_relationships = []
        
        if not relationships_df.empty and 'part_id' in relationships_df.columns:
            # Filter relationships matching this part_id
            matching_rels = relationships_df[relationships_df['part_id'] == part_id]
            
            for _, rel_row in matching_rels.iterrows():
                rel_dict = rel_row.to_dict()
                supplier_id = rel_dict.get('supplier_id')
                
                # Get full supplier details
                if supplier_id and supplier_id in suppliers_dict:
                    supplier_details = suppliers_dict[supplier_id].copy()
                    # Combine relationship data with supplier details
                    enriched_rel = {
                        **rel_dict,
                        'supplier_details': supplier_details
                    }
                    part_relationships.append(enriched_rel)
        
        # Add suppliers field to part
        enriched_part['suppliers'] = part_relationships
        
        # Ensure oem_id is present (CRITICAL for Qdrant 1.16 multitenancy)
        # Distribute evenly across OEM-A, OEM-B, OEM-C
        if 'oem_id' not in enriched_part or not enriched_part.get('oem_id'):
            oem_index = idx % len(oem_ids)
            enriched_part['oem_id'] = oem_ids[oem_index]
        
        # Create search_text field by concatenating relevant fields
        search_text_parts = []
        
        # Add part name
        if part.get('part_name'):
            search_text_parts.append(str(part['part_name']))
        
        # Add description
        if part.get('description'):
            search_text_parts.append(str(part['description']))
        
        # Add specifications
        if part.get('specifications'):
            search_text_parts.append(str(part['specifications']))
        
        # Add category
        if part.get('category'):
            search_text_parts.append(str(part['category']))
        
        # Add all supplier names
        supplier_names = []
        for rel in part_relationships:
            supplier_details = rel.get('supplier_details', {})
            if supplier_details.get('company_name'):
                supplier_names.append(supplier_details['company_name'])
        
        if supplier_names:
            search_text_parts.extend(supplier_names)
        
        # Add compliance standards (for full-text search demo)
        if part.get('compliance_standards'):
            if isinstance(part['compliance_standards'], list):
                search_text_parts.extend([str(std) for std in part['compliance_standards']])
            else:
                search_text_parts.append(str(part['compliance_standards']))
        
        # Join all parts with spaces
        enriched_part['search_text'] = ' '.join(search_text_parts)
        
        # Add normalized_part_name for full-text search testing
        # Normalize: lowercase, remove special characters (keep spaces and alphanumeric)
        if part.get('part_name'):
            normalized = str(part['part_name']).lower()
            # Remove special characters but keep spaces, numbers, and letters
            normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in normalized)
            # Collapse multiple spaces
            normalized = ' '.join(normalized.split())
            enriched_part['normalized_part_name'] = normalized
        else:
            enriched_part['normalized_part_name'] = ''
        
        enriched_parts.append(enriched_part)
        
        # Log progress every 100 parts
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{total_parts} parts...")
    
    # Log summary statistics
    total_suppliers_linked = sum(len(part.get('suppliers', [])) for part in enriched_parts)
    parts_with_suppliers = sum(1 for part in enriched_parts if part.get('suppliers'))
    
    logger.success(f"✅ Enrichment complete!")
    logger.info(f"  Total parts enriched: {len(enriched_parts)}")
    logger.info(f"  Parts with suppliers: {parts_with_suppliers}")
    logger.info(f"  Total supplier relationships: {total_suppliers_linked}")
    logger.info(f"  Parts with oem_id: {sum(1 for p in enriched_parts if p.get('oem_id'))}")
    logger.info(f"  Parts with search_text: {sum(1 for p in enriched_parts if p.get('search_text'))}")
    
    return enriched_parts

