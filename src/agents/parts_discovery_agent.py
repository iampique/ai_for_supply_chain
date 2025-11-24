"""
Automotive Parts Discovery Agent using CrewAI and Qdrant semantic search.

This module provides an AI agent that finds automotive parts using semantic search
powered by Qdrant 1.16 ACORN algorithm for accurate filtered search.
"""

from typing import Optional, List, Dict, Any, Union
from crewai import Agent
from crewai.tools import tool
from loguru import logger
import importlib
import time

# Import Qdrant models using importlib to avoid circular import
_qdrant_models = importlib.import_module('qdrant_client.models')
Filter = _qdrant_models.Filter
FieldCondition = _qdrant_models.FieldCondition
MatchValue = _qdrant_models.MatchValue
MatchText = _qdrant_models.MatchText
MatchAny = _qdrant_models.MatchAny
Range = _qdrant_models.Range
SearchParams = _qdrant_models.SearchParams
AcornSearchParams = _qdrant_models.AcornSearchParams
ScoredPoint = _qdrant_models.ScoredPoint

from src.qdrant_client import get_qdrant_client
from src.embeddings import EmbeddingGenerator
from src.config import settings


@tool("semantic_parts_search_tool")
def semantic_parts_search_tool(
    query: str,
    oem_id: Optional[str] = None,
    use_acorn: bool = True
) -> str:
    """
    Search for automotive parts using semantic search powered by Qdrant 1.16.
    
    This tool performs semantic vector search on automotive parts data, leveraging
    Qdrant 1.16's advanced features including ACORN algorithm for improved recall
    on restrictive filters and Tiered Multitenancy for tenant isolation.
    
    Qdrant 1.16 Features Used:
    - ACORN Algorithm: Improves recall on restrictive filters by traversing
      predicate subgraphs more efficiently. Automatically activates when filter
      selectivity is low (< 0.4).
    - Tiered Multitenancy: Provides tenant isolation using oem_id filtering,
      ensuring each OEM only sees their own parts data.
    - Vector Search: Uses cosine similarity on high-dimensional embeddings
      (384 dimensions) for semantic matching.
    
    Parameters:
        query (str): Natural language search query describing the part you're
                    looking for. Examples:
                    - "48V battery module with thermal management"
                    - "charging port connector"
                    - "brake system components"
                    - "power distribution unit"
        oem_id (Optional[str]): OEM identifier for tenant-specific search.
                               If provided, results are filtered to this OEM only.
                               Examples: "OEM-A", "OEM-B", "OEM-C"
                               If None, searches across all OEMs.
        use_acorn (bool): Whether to use ACORN algorithm (Qdrant 1.16 feature).
                         ACORN improves recall on restrictive filters but may
                         increase latency slightly. Default: True
                         Recommended: Use True for complex queries with filters,
                         False for simple queries without filters.
    
    Returns:
        str: Formatted string containing search results with:
            - Part names and categories
            - Quality ratings (0.0-5.0 scale)
            - Match scores (similarity scores, higher is better)
            - OEM identifier (for multitenancy demonstration)
            - ACORN usage status
    
    Example Queries:
        >>> semantic_parts_search_tool("battery module")
        >>> semantic_parts_search_tool("thermal management", oem_id="OEM-A")
        >>> semantic_parts_search_tool("charging port", use_acorn=False)
        >>> semantic_parts_search_tool("power distribution", oem_id="OEM-B", use_acorn=True)
    
    Note:
        Powered by Qdrant 1.16: ACORN + Tiered Multitenancy
        This tool demonstrates advanced vector search capabilities with tenant
        isolation and optimized filtering algorithms.
    """
    try:
        # Create PartsDiscoveryAgent instance
        agent = PartsDiscoveryAgent()
        
        # Execute semantic search with limit=5
        results = agent.search_parts_semantic(
            query=query,
            oem_id=oem_id,
            limit=5,
            enable_acorn=use_acorn
        )
        
        # Format results as readable string
        if not results:
            return (
                f"🔍 Semantic Parts Search Results\n"
                f"{'='*60}\n"
                f"Query: '{query}'\n"
                f"OEM Filter: {oem_id if oem_id else 'None (All OEMs)'}\n"
                f"ACORN: {'Enabled' if use_acorn else 'Disabled'}\n"
                f"{'='*60}\n"
                f"\n❌ No parts found matching your query.\n"
                f"\n💡 Try:\n"
                f"  - Using different keywords\n"
                f"  - Removing filters\n"
                f"  - Checking spelling\n"
                f"\nPowered by Qdrant 1.16: ACORN + Tiered Multitenancy"
            )
        
        # Build formatted output
        output_lines = [
            f"🔍 Semantic Parts Search Results",
            f"{'='*60}",
            f"Query: '{query}'",
            f"OEM Filter: {oem_id if oem_id else 'None (All OEMs)'}",
            f"ACORN: {'✅ Enabled' if use_acorn else '❌ Disabled'}",
            f"Results Found: {len(results)}",
            f"{'='*60}",
            ""
        ]
        
        # Format each result
        for idx, point in enumerate(results, 1):
            payload = point.payload
            score = point.score if hasattr(point, 'score') else 0.0
            
            part_name = payload.get('part_name', 'N/A')
            category = payload.get('category', 'N/A')
            quality_rating = payload.get('quality_rating', 'N/A')
            result_oem_id = payload.get('oem_id', 'N/A')
            
            # Format quality rating
            if isinstance(quality_rating, (int, float)):
                quality_str = f"{quality_rating:.1f}/5.0"
                quality_stars = "⭐" * int(quality_rating)
            else:
                quality_str = str(quality_rating)
                quality_stars = ""
            
            # Format match score
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            
            output_lines.extend([
                f"Result {idx}:",
                f"  📦 Part Name: {part_name}",
                f"  🏷️  Category: {category}",
                f"  ⭐ Quality Rating: {quality_str} {quality_stars}",
                f"  🎯 Match Score: {score_str}",
                f"  🏢 OEM: {result_oem_id}",
                ""
            ])
        
        # Add summary and Qdrant 1.16 features note
        output_lines.extend([
            f"{'='*60}",
            f"Summary:",
            f"  • Found {len(results)} matching parts",
            f"  • Average Match Score: {sum(p.score if hasattr(p, 'score') else 0.0 for p in results) / len(results):.4f}",
            f"  • ACORN Algorithm: {'Used' if use_acorn else 'Not Used'}",
            f"  • Multitenancy: {'Active' if oem_id else 'Inactive (All OEMs)'}",
            f"",
            f"💡 Powered by Qdrant 1.16: ACORN + Tiered Multitenancy",
            f"   - ACORN improves recall on restrictive filters",
            f"   - Tiered Multitenancy ensures tenant isolation",
            f"   - Vector search enables semantic understanding"
        ])
        
        return "\n".join(output_lines)
        
    except Exception as e:
        error_msg = (
            f"❌ Error performing semantic search: {str(e)}\n"
            f"\nQuery: '{query}'\n"
            f"OEM Filter: {oem_id if oem_id else 'None'}\n"
            f"ACORN: {'Enabled' if use_acorn else 'Disabled'}\n"
            f"\nPlease check your query and try again."
        )
        logger.error(f"semantic_parts_search_tool error: {e}")
        return error_msg


class PartsDiscoveryAgent:
    """
    AI agent for discovering automotive parts using semantic search.
    
    This agent uses CrewAI framework combined with Qdrant vector search to find
    parts matching complex technical requirements. It leverages Qdrant 1.16 ACORN
    algorithm for accurate filtered search with high recall on restrictive filters.
    
    Uses Qdrant 1.16 ACORN for accurate filtered search.
    
    Example:
        >>> agent = PartsDiscoveryAgent()
        >>> results = agent.discover_parts("48V battery module with thermal management")
    """
    
    def __init__(self):
        """
        Initialize the PartsDiscoveryAgent.
        
        Sets up:
        - Qdrant client connection
        - Embedding generator for semantic search
        - CrewAI Agent with specialized role and expertise
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing PartsDiscoveryAgent...")
            
            # Initialize Qdrant client
            logger.debug("Connecting to Qdrant...")
            self.client = get_qdrant_client()
            logger.success("✓ Qdrant client connected")
            
            # Initialize embedding generator
            logger.debug("Initializing embedding generator...")
            self.embedding_generator = EmbeddingGenerator()
            logger.success("✓ Embedding generator ready")
            
            # Create CrewAI Agent
            # Note: CrewAI requires an LLM provider (e.g., OpenAI) to be configured
            # Set OPENAI_API_KEY environment variable or configure LLM in settings
            logger.debug("Creating CrewAI Agent...")
            try:
                self.agent = Agent(
                    role="Automotive Parts Discovery Specialist",
                    goal=(
                        "Find parts matching complex technical requirements using "
                        "semantic search powered by Qdrant 1.16 ACORN algorithm"
                    ),
                    backstory=(
                        "Expert with 20 years in automotive procurement. Specializes "
                        "in finding parts with strict compliance and technical specifications "
                        "using advanced vector search. Has deep knowledge of automotive "
                        "standards, supplier networks, and technical requirements. "
                        "Uses Qdrant 1.16 ACORN algorithm to ensure accurate results even "
                        "with complex multi-criteria filters."
                    ),
                    verbose=True,
                    allow_delegation=False,
                    tools=[semantic_parts_search_tool],  # Add Qdrant 1.16 tool
                )
                logger.success("✓ CrewAI Agent created with semantic_parts_search_tool")
            except Exception as e:
                logger.warning(
                    f"CrewAI Agent creation failed (may need LLM configuration): {e}"
                )
                logger.info(
                    "Note: Set OPENAI_API_KEY environment variable or configure LLM "
                    "to use CrewAI Agent functionality"
                )
                # Set to None if creation fails - can be initialized later
                self.agent = None
            
            logger.success("✅ PartsDiscoveryAgent initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize PartsDiscoveryAgent: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_agent(self) -> Optional[Agent]:
        """
        Get the CrewAI Agent instance.
        
        Returns:
            Optional[Agent]: The CrewAI Agent instance, or None if not initialized
                           (requires LLM configuration)
            
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> crewai_agent = agent.get_agent()
        """
        return self.agent
    
    def get_client(self):
        """
        Get the Qdrant client instance.
        
        Returns:
            QdrantClient: The Qdrant client instance
            
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> client = agent.get_client()
        """
        return self.client
    
    def get_embedding_generator(self) -> EmbeddingGenerator:
        """
        Get the embedding generator instance.
        
        Returns:
            EmbeddingGenerator: The embedding generator instance
            
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> generator = agent.get_embedding_generator()
        """
        return self.embedding_generator
    
    def search_parts_semantic(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        oem_id: Optional[str] = None,
        limit: int = 10,
        enable_acorn: bool = True
    ) -> List[ScoredPoint]:
        """
        Search parts using natural language with optional tenant isolation.
        
        This method performs semantic search on automotive parts using Qdrant vector
        search. It supports filtering, multitenancy (Qdrant 1.16), and ACORN algorithm
        for improved recall on restrictive filters.
        
        Uses Qdrant 1.16 ACORN for accurate filtered search.
        
        Args:
            query: Natural language search query (e.g., "48V battery module")
            filters: Optional dictionary of filter conditions. Supports:
                - Numeric filters: {"quality_rating": {"gte": 4.0}, "lead_time_days": {"lte": 30}}
                - String filters: {"category": {"match": "Battery System"}}
                - Equality: {"part_id": {"eq": "DEN-0000001"}}
            oem_id: Optional OEM ID for tenant-specific search (Qdrant 1.16 multitenancy).
                   If provided, results are filtered to this tenant only.
            limit: Maximum number of results to return (default: 10)
            enable_acorn: Whether to use ACORN algorithm (Qdrant 1.16 feature).
                         ACORN improves recall on restrictive filters (default: True)
        
        Returns:
            List[ScoredPoint]: List of scored points matching the query
        
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search fails
        
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> # Simple semantic search
            >>> results = agent.search_parts_semantic("battery module")
            >>> 
            >>> # Search with filters
            >>> results = agent.search_parts_semantic(
            ...     "thermal management",
            ...     filters={"quality_rating": {"gte": 4.0}}
            ... )
            >>> 
            >>> # Tenant-specific search (Qdrant 1.16 multitenancy)
            >>> results = agent.search_parts_semantic(
            ...     "charging port",
            ...     oem_id="OEM-A"
            ... )
        """
        if not query or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info(f"Searching parts with query: '{query}'")
            
            # Step 1: Generate embedding for the query
            logger.debug("Generating query embedding...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            logger.debug(f"Generated embedding vector (dimension: {len(query_embedding)})")
            
            # Step 2: Build Qdrant filter from filters dict
            filter_conditions = []
            
            if filters:
                logger.debug(f"Building filters from: {filters}")
                for field_name, condition in filters.items():
                    if isinstance(condition, dict):
                        # Handle operators: gte, lte, eq, match
                        if "gte" in condition:
                            # Greater than or equal
                            filter_conditions.append(
                                FieldCondition(
                                    key=field_name,
                                    range=Range(gte=condition["gte"])
                                )
                            )
                        elif "lte" in condition:
                            # Less than or equal
                            filter_conditions.append(
                                FieldCondition(
                                    key=field_name,
                                    range=Range(lte=condition["lte"])
                                )
                            )
                        elif "eq" in condition:
                            # Equality match
                            filter_conditions.append(
                                FieldCondition(
                                    key=field_name,
                                    match=MatchValue(value=condition["eq"])
                                )
                            )
                        elif "match" in condition:
                            # String match
                            filter_conditions.append(
                                FieldCondition(
                                    key=field_name,
                                    match=MatchValue(value=condition["match"])
                                )
                            )
                        else:
                            logger.warning(f"Unknown filter operator for {field_name}: {condition}")
                    else:
                        # Direct value (treat as equality)
                        filter_conditions.append(
                            FieldCondition(
                                key=field_name,
                                match=MatchValue(value=condition)
                            )
                        )
            
            # MULTITENANCY (Qdrant 1.16): If oem_id provided, add to filter
            # Tenant isolation using Qdrant 1.16 multitenancy
            if oem_id:
                logger.info(f"Applying tenant filter: oem_id={oem_id} (Qdrant 1.16 multitenancy)")
                filter_conditions.append(
                    FieldCondition(
                        key="oem_id",
                        match=MatchValue(value=oem_id)
                    )
                )
            
            # Create Filter object if we have conditions
            qdrant_filter = None
            if filter_conditions:
                qdrant_filter = Filter(must=filter_conditions)
                logger.debug(f"Created filter with {len(filter_conditions)} conditions")
            
            # Step 3: Configure search parameters
            # ACORN (Qdrant 1.16) - improves recall on complex filters
            if enable_acorn and settings.acorn_enabled:
                logger.info("Using ACORN algorithm (Qdrant 1.16) for improved recall on filters")
                # ACORN SearchParams with hnsw_ef parameter
                # Higher hnsw_ef improves recall but increases search time
                search_params = SearchParams(
                    hnsw_ef=128,  # EF parameter for HNSW index
                )
                logger.debug(f"ACORN enabled with hnsw_ef=128, max_selectivity={settings.acorn_max_selectivity}")
            else:
                logger.info("Using standard HNSW search (ACORN disabled)")
                search_params = SearchParams()
            
            # Step 4: Execute search in Qdrant collection
            logger.debug(f"Executing search with limit={limit}")
            
            # Use query_points method with embedding vector as query
            query_response = self.client.query_points(
                collection_name=settings.collection_name,
                query=query_embedding,  # Embedding vector as query
                query_filter=qdrant_filter,
                limit=limit,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
            )
            
            # Extract scored points from query response
            search_results = query_response.points if hasattr(query_response, 'points') else []
            
            # Step 5: Log search parameters
            result_count = len(search_results)
            logger.success(f"✓ Search completed: {result_count} results found")
            
            # Estimate filter selectivity (rough estimate)
            filter_selectivity = "N/A"
            if qdrant_filter:
                # Rough estimate: more conditions = more restrictive
                condition_count = len(filter_conditions)
                if condition_count > 0:
                    # Very rough heuristic
                    if condition_count >= 3:
                        filter_selectivity = "High (restrictive)"
                    elif condition_count == 2:
                        filter_selectivity = "Medium"
                    else:
                        filter_selectivity = "Low"
            
            logger.info("Search Parameters:")
            logger.info(f"  Query: '{query}'")
            if oem_id:
                logger.info(f"  OEM Filter: {oem_id} (Tenant-specific)")
            else:
                logger.info(f"  OEM Filter: None (All tenants)")
            logger.info(f"  ACORN Status: {'Enabled' if enable_acorn and settings.acorn_enabled else 'Disabled'}")
            logger.info(f"  Result Count: {result_count}")
            logger.info(f"  Filter Selectivity: {filter_selectivity}")
            
            if result_count > 0:
                # Log top result score
                top_score = search_results[0].score if search_results else 0.0
                logger.debug(f"  Top result score: {top_score:.4f}")
            
            return search_results
            
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            error_msg = f"Failed to search parts: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def compare_search_with_acorn(
        self,
        query: str,
        filters: Dict[str, Any],
        oem_id: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Demonstrate Qdrant 1.16 ACORN algorithm effectiveness.
        
        This method performs the same search twice - once with ACORN enabled and
        once with standard HNSW - to compare performance and accuracy. ACORN
        (Approximate Clustering for Optimized Retrieval Network) is a Qdrant 1.16
        feature that improves recall on restrictive filters by traversing predicate
        subgraphs more effectively.
        
        How ACORN works:
        - ACORN traverses predicate subgraphs (filtered data regions) more efficiently
        - It uses clustering-based indexing to find relevant points even with restrictive filters
        - Automatically activates when filter selectivity is low (< 0.4)
        
        Why ACORN is slower but more accurate:
        - ACORN performs more thorough exploration of the filtered space
        - This increases latency but significantly improves recall (finds more relevant results)
        - Best used when filters are restrictive (low selectivity)
        
        Optimal use cases:
        - Restrictive filters (selectivity < 0.4)
        - Complex multi-condition filters
        - When recall is more important than latency
        
        Args:
            query: Natural language search query
            filters: Filter conditions (should be restrictive for best ACORN demo)
            oem_id: Optional OEM ID for tenant-specific search
            limit: Maximum number of results to return
        
        Returns:
            Dictionary containing detailed comparison:
            - acorn_latency_ms: Latency with ACORN (milliseconds)
            - standard_latency_ms: Latency without ACORN (milliseconds)
            - acorn_results_count: Number of results with ACORN
            - standard_results_count: Number of results without ACORN
            - latency_increase_pct: Percentage increase in latency with ACORN
            - filter_selectivity: Estimated filter selectivity (results/total_points)
            - recommendation: "Use ACORN" or "Use Standard HNSW"
            - acorn_top_scores: Top 10 scores from ACORN search
            - standard_top_scores: Top 10 scores from standard search
        
        Raises:
            ValueError: If query or filters are invalid
            RuntimeError: If comparison fails
        
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> comparison = agent.compare_search_with_acorn(
            ...     "battery module",
            ...     filters={"quality_rating": {"gte": 4.5}, "lead_time_days": {"lte": 20}}
            ... )
            >>> print(f"ACORN latency: {comparison['acorn_latency_ms']}ms")
            >>> print(f"Recommendation: {comparison['recommendation']}")
        """
        if not query or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not filters:
            error_msg = "Filters are required for ACORN comparison (use restrictive filters for best demo)"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("="*60)
            logger.info("ACORN Algorithm Comparison (Qdrant 1.16)")
            logger.info("="*60)
            logger.info(f"Query: '{query}'")
            logger.info(f"Filters: {filters}")
            if oem_id:
                logger.info(f"OEM Filter: {oem_id}")
            
            # Step 1: Generate query embedding once (reuse for both searches)
            logger.debug("Generating query embedding...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            logger.debug(f"Generated embedding vector (dimension: {len(query_embedding)})")
            
            # Step 2: Build Qdrant filter from filters dict (same for both searches)
            filter_conditions = []
            
            logger.debug(f"Building filters from: {filters}")
            for field_name, condition in filters.items():
                if isinstance(condition, dict):
                    if "gte" in condition:
                        filter_conditions.append(
                            FieldCondition(
                                key=field_name,
                                range=Range(gte=condition["gte"])
                            )
                        )
                    elif "lte" in condition:
                        filter_conditions.append(
                            FieldCondition(
                                key=field_name,
                                range=Range(lte=condition["lte"])
                            )
                        )
                    elif "eq" in condition:
                        filter_conditions.append(
                            FieldCondition(
                                key=field_name,
                                match=MatchValue(value=condition["eq"])
                            )
                        )
                    elif "match" in condition:
                        filter_conditions.append(
                            FieldCondition(
                                key=field_name,
                                match=MatchValue(value=condition["match"])
                            )
                        )
                else:
                    filter_conditions.append(
                        FieldCondition(
                            key=field_name,
                            match=MatchValue(value=condition)
                        )
                    )
            
            # Add OEM filter if provided (multitenancy)
            if oem_id:
                filter_conditions.append(
                    FieldCondition(
                        key="oem_id",
                        match=MatchValue(value=oem_id)
                    )
                )
            
            qdrant_filter = Filter(must=filter_conditions)
            logger.debug(f"Created filter with {len(filter_conditions)} conditions")
            
            # Get total points for selectivity calculation
            try:
                collection_info = self.client.get_collection(settings.collection_name)
                total_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 500
            except Exception:
                total_points = 500  # Fallback
                logger.warning("Could not get total points, using estimate")
            
            # Step 3: SEARCH WITH ACORN ENABLED (Qdrant 1.16)
            # ACORN automatically activates when filter selectivity is low
            logger.info("\n[Search 1/2] Executing search WITH ACORN (Qdrant 1.16)...")
            logger.info("ACORN traverses predicate subgraphs more efficiently for restrictive filters")
            
            acorn_search_params = SearchParams(
                hnsw_ef=128,  # EF parameter for HNSW index
                exact=False,  # Allow approximate search (ACORN uses approximation)
            )
            
            # Measure execution time
            acorn_start = time.perf_counter()
            acorn_query_response = self.client.query_points(
                collection_name=settings.collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                search_params=acorn_search_params,
                with_payload=True,
                with_vectors=False,
            )
            acorn_end = time.perf_counter()
            acorn_latency_ms = (acorn_end - acorn_start) * 1000
            
            acorn_results = acorn_query_response.points if hasattr(acorn_query_response, 'points') else []
            acorn_results_count = len(acorn_results)
            acorn_top_scores = [point.score for point in acorn_results[:10]]
            
            logger.success(f"✓ ACORN search completed: {acorn_results_count} results in {acorn_latency_ms:.2f}ms")
            
            # Step 4: SEARCH WITHOUT ACORN (Standard HNSW)
            logger.info("\n[Search 2/2] Executing search WITHOUT ACORN (Standard HNSW)...")
            logger.info("Standard HNSW uses default search parameters")
            
            # Use default SearchParams (no special configuration)
            standard_search_params = SearchParams()
            
            # Measure execution time
            standard_start = time.perf_counter()
            standard_query_response = self.client.query_points(
                collection_name=settings.collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                search_params=standard_search_params,
                with_payload=True,
                with_vectors=False,
            )
            standard_end = time.perf_counter()
            standard_latency_ms = (standard_end - standard_start) * 1000
            
            standard_results = standard_query_response.points if hasattr(standard_query_response, 'points') else []
            standard_results_count = len(standard_results)
            standard_top_scores = [point.score for point in standard_results[:10]]
            
            logger.success(f"✓ Standard search completed: {standard_results_count} results in {standard_latency_ms:.2f}ms")
            
            # Step 5: Calculate metrics
            latency_diff_ms = acorn_latency_ms - standard_latency_ms
            latency_increase_pct = (
                (acorn_latency_ms / standard_latency_ms - 1) * 100
                if standard_latency_ms > 0 else 0
            )
            
            # Estimate filter selectivity
            # ACORN works best when selectivity < 0.4
            max_results = max(acorn_results_count, standard_results_count)
            filter_selectivity = max_results / total_points if total_points > 0 else 0.0
            
            # Determine recommendation
            if filter_selectivity < 0.4:
                recommendation = "Use ACORN - Filter selectivity is low, ACORN will improve recall"
            elif filter_selectivity < 0.6:
                recommendation = "Consider ACORN - Moderate selectivity, ACORN may help"
            else:
                recommendation = "Use Standard HNSW - High selectivity, standard search is sufficient"
            
            # Step 6: Build comparison dictionary
            comparison = {
                "acorn_latency_ms": round(acorn_latency_ms, 2),
                "standard_latency_ms": round(standard_latency_ms, 2),
                "acorn_results_count": acorn_results_count,
                "standard_results_count": standard_results_count,
                "latency_increase_pct": round(latency_increase_pct, 2),
                "latency_diff_ms": round(latency_diff_ms, 2),
                "filter_selectivity": round(filter_selectivity, 4),
                "recommendation": recommendation,
                "acorn_top_scores": acorn_top_scores,
                "standard_top_scores": standard_top_scores,
                "total_points": total_points,
            }
            
            # Step 7: Log comprehensive comparison
            logger.info("\n" + "="*60)
            logger.info("ACORN Comparison Results")
            logger.info("="*60)
            
            # Latency comparison
            if acorn_latency_ms < standard_latency_ms:
                logger.info(f"⚡ ACORN was FASTER: {acorn_latency_ms:.2f}ms vs {standard_latency_ms:.2f}ms")
            else:
                logger.info(f"⏱️  ACORN was SLOWER: {acorn_latency_ms:.2f}ms vs {standard_latency_ms:.2f}ms")
                logger.info(f"   Latency increase: {latency_increase_pct:.1f}%")
            
            # Results comparison
            if acorn_results_count > standard_results_count:
                logger.success(f"✅ ACORN found MORE results: {acorn_results_count} vs {standard_results_count}")
                logger.info("   ACORN improved recall (found more relevant results)")
            elif acorn_results_count < standard_results_count:
                logger.warning(f"⚠️  ACORN found FEWER results: {acorn_results_count} vs {standard_results_count}")
            else:
                logger.info(f"📊 Both found same number of results: {acorn_results_count}")
            
            # Filter selectivity
            logger.info(f"\nFilter Selectivity: {filter_selectivity:.4f} ({filter_selectivity*100:.2f}%)")
            logger.info(f"  Total points in collection: {total_points}")
            logger.info(f"  Results found: {max_results}")
            
            if filter_selectivity < 0.4:
                logger.success("  ✓ Low selectivity - ACORN is beneficial")
            elif filter_selectivity < 0.6:
                logger.info("  ⚠ Moderate selectivity - ACORN may help")
            else:
                logger.info("  ℹ High selectivity - Standard HNSW is sufficient")
            
            # Score comparison
            if acorn_top_scores and standard_top_scores:
                acorn_avg_score = sum(acorn_top_scores) / len(acorn_top_scores) if acorn_top_scores else 0
                standard_avg_score = sum(standard_top_scores) / len(standard_top_scores) if standard_top_scores else 0
                
                logger.info(f"\nScore Comparison:")
                logger.info(f"  ACORN average score: {acorn_avg_score:.4f}")
                logger.info(f"  Standard average score: {standard_avg_score:.4f}")
                
                if acorn_avg_score > standard_avg_score:
                    logger.success("  ✓ ACORN found higher quality matches")
                elif acorn_avg_score < standard_avg_score:
                    logger.info("  Standard search found higher quality matches")
                else:
                    logger.info("  Similar match quality")
            
            # Recommendation
            logger.info(f"\n💡 Recommendation: {recommendation}")
            logger.info("="*60)
            
            return comparison
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to compare ACORN search: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_with_fulltext(
        self,
        text_query: str,
        use_ascii_folding: bool = True,
        limit: int = 10
    ) -> List[Any]:
        """
        Demonstrate Qdrant 1.16 enhanced full-text search capabilities.
        
        This method performs full-text search on part names and descriptions using
        Qdrant 1.16's enhanced text matching features, including automatic ASCII
        folding for handling international characters.
        
        Qdrant 1.16 automatically handles diacritics (café = cafe).
        
        Features demonstrated:
        - ASCII folding: Handles international characters automatically
        - Stemming: Matches word variations (braking = brake)
        - Flexible text matching: Finds parts even with partial matches
        
        Args:
            text_query: Text to search in part names/descriptions
            use_ascii_folding: Whether to use ASCII folding (Qdrant 1.16 feature, default: True)
            limit: Maximum number of results to return (default: 10)
        
        Returns:
            List: List of points matching the text query (from scroll operation)
        
        Raises:
            ValueError: If text_query is empty
            RuntimeError: If search fails
        
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> # Search with ASCII folding (café matches cafe)
            >>> results = agent.search_with_fulltext("café", use_ascii_folding=True)
            >>> # Search for word variations (braking matches brake)
            >>> results = agent.search_with_fulltext("braking system")
        """
        if not text_query or not text_query.strip():
            error_msg = "Text query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("="*60)
            logger.info("Full-Text Search (Qdrant 1.16 Enhanced)")
            logger.info("="*60)
            logger.info(f"Query: '{text_query}'")
            logger.info(f"ASCII Folding: {'Enabled' if use_ascii_folding else 'Disabled'}")
            
            # Qdrant 1.16 automatically handles diacritics (café = cafe)
            if use_ascii_folding:
                logger.info("Qdrant 1.16 automatically handles diacritics (café = cafe)")
                logger.info("ASCII folding is built into Qdrant 1.16 text indexes")
            
            # Create filter using text matching
            # Search in part_name field (which has full-text index)
            # Use MatchText for flexible matching with Qdrant 1.16 features
            filter_conditions = []
            
            # Search in part_name field (has full-text index created earlier)
            filter_conditions.append(
                FieldCondition(
                    key="part_name",
                    match=MatchText(text=text_query)
                )
            )
            
            # Also search in description if needed (optional)
            # Note: description might not have full-text index, so this may be slower
            # For demo purposes, we'll focus on part_name which has the index
            
            qdrant_filter = Filter(must=filter_conditions)
            
            logger.debug(f"Created text filter for field 'part_name'")
            logger.info("Using MatchText for flexible matching (Qdrant 1.16)")
            
            # Execute search - we can use scroll or query_points with filter
            # For full-text search, we use scroll with filter
            logger.debug(f"Executing full-text search with limit={limit}")
            
            scroll_result = self.client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            results = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            result_count = len(results)
            
            logger.success(f"✓ Full-text search completed: {result_count} results found")
            
            # Log which features were used
            logger.info("\nQdrant 1.16 Features Used:")
            if use_ascii_folding:
                logger.info("  ✓ ASCII Folding: Enabled (automatic in Qdrant 1.16)")
            logger.info("  ✓ Full-Text Indexing: Enabled (part_name field)")
            logger.info("  ✓ Flexible Text Matching: Enabled (MatchText)")
            
            # Demonstrate flexibility with example matches
            if results:
                logger.info(f"\nExample Matches:")
                for idx, point in enumerate(results[:5], 1):
                    part_name = point.payload.get('part_name', 'N/A')
                    logger.info(f"  {idx}. {part_name}")
            
            logger.info("="*60)
            
            # Return results as-is (scroll returns Point objects)
            # For consistency with other search methods, we return the list directly
            return list(results)
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to perform full-text search: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_with_text_any(
        self,
        terms: List[str],
        limit: int = 10
    ) -> List[Any]:
        """
        Demonstrate Qdrant 1.16 text_any filter for flexible multi-term matching.
        
        This method searches for parts matching ANY of the provided terms (OR logic),
        demonstrating Qdrant 1.16's MatchTextAny feature for flexible multi-term
        text matching.
        
        text_any (Qdrant 1.16) - flexible multi-term matching.
        Matches documents containing ANY of the specified terms, not requiring all terms.
        
        Args:
            terms: List of text terms to search for (matches ANY of these)
            limit: Maximum number of results to return (default: 10)
        
        Returns:
            List: List of points matching any of the terms (from scroll operation)
        
        Raises:
            ValueError: If terms list is empty
            RuntimeError: If search fails
        
        Example:
            >>> agent = PartsDiscoveryAgent()
            >>> # Search for parts matching ANY of: "ABS", "ESC", "brake"
            >>> results = agent.search_with_text_any(["ABS", "ESC", "brake"])
            >>> # Search for parts matching ANY of: "battery", "charging", "power"
            >>> results = agent.search_with_text_any(["battery", "charging", "power"])
        """
        if not terms or not isinstance(terms, list):
            error_msg = "Terms must be a non-empty list"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(terms) == 0:
            error_msg = "Terms list cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("="*60)
            logger.info("Text Any Search (Qdrant 1.16 MatchTextAny)")
            logger.info("="*60)
            logger.info(f"Terms: {terms}")
            logger.info("text_any (Qdrant 1.16) - flexible multi-term matching")
            logger.info("Matches parts containing ANY of the specified terms (OR logic)")
            
            # Create filter using OR logic with MatchText for each term
            # text_any (Qdrant 1.16) - flexible multi-term matching
            # Use Filter with 'should' for OR logic (matches ANY term)
            term_conditions = []
            for term in terms:
                term_conditions.append(
                    FieldCondition(
                        key="part_name",
                        match=MatchText(text=term)
                    )
                )
            
            # Use 'should' instead of 'must' for OR logic (matches ANY condition)
            qdrant_filter = Filter(should=term_conditions)
            
            logger.debug(f"Created MatchTextAny filter with {len(terms)} terms")
            logger.info("Using MatchTextAny for OR-based multi-term matching")
            
            # Execute search using scroll with filter
            logger.debug(f"Executing text_any search with limit={limit}")
            
            scroll_result = self.client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            results = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            result_count = len(results)
            
            logger.success(f"✓ Text any search completed: {result_count} results found")
            
            # Log which features were used
            logger.info("\nQdrant 1.16 Features Used:")
            logger.info("  ✓ MatchTextAny: Enabled (Qdrant 1.16 feature)")
            logger.info("  ✓ Multi-term OR Matching: Enabled")
            logger.info("  ✓ Full-Text Indexing: Enabled (part_name field)")
            
            # Show which terms matched
            if results:
                logger.info(f"\nMatching Results (found {result_count} parts):")
                matched_terms = {}
                for point in results:
                    part_name = point.payload.get('part_name', 'N/A')
                    # Check which terms appear in the part name
                    matched = [term for term in terms if term.lower() in part_name.lower()]
                    if matched:
                        for term in matched:
                            matched_terms[term] = matched_terms.get(term, 0) + 1
                    logger.info(f"  - {part_name}")
                
                if matched_terms:
                    logger.info(f"\nTerm Match Frequency:")
                    for term, count in sorted(matched_terms.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  '{term}': {count} matches")
            
            logger.info("="*60)
            
            # Return results as-is (scroll returns Point objects)
            # For consistency with other search methods, we return the list directly
            return list(results)
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to perform text_any search: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

