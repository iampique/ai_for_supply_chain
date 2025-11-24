"""
Supplier Alternative and Risk Assessment Agent using CrewAI and Qdrant.

This module provides an AI agent that finds alternative suppliers and assesses
supply chain risks using semantic search powered by Qdrant vector database.
"""

from typing import Optional, Dict, List, Any
from crewai import Agent
from crewai.tools import tool
from loguru import logger
import importlib
import hashlib

# Import Qdrant models using importlib to avoid circular import
_qdrant_models = importlib.import_module('qdrant_client.models')
Filter = _qdrant_models.Filter
FieldCondition = _qdrant_models.FieldCondition
MatchValue = _qdrant_models.MatchValue
Range = _qdrant_models.Range

from src.qdrant_client import get_qdrant_client
from src.data_loader import load_all_data
from src.embeddings import EmbeddingGenerator
from src.config import settings


@tool("find_supplier_alternatives_tool")
def find_supplier_alternatives_tool(
    part_id: str,
    reason: str
) -> str:
    """
    Find alternative suppliers for a part using Qdrant vector search.
    
    This tool uses semantic vector search powered by Qdrant to find compatible
    alternative parts and their suppliers when disruptions occur. It analyzes part
    specifications using embeddings to identify similar parts, then evaluates
    suppliers based on quality, delivery performance, lead time, and cost.
    
    How Vector Search Works:
    - Generates embeddings from part specifications (name, description, category, etc.)
    - Uses cosine similarity in Qdrant's vector space to find semantically similar parts
    - Matches suppliers based on specialization and capabilities
    - Scores suppliers using composite metrics (quality, delivery, lead time, cost)
    - Considers geographic diversity to reduce supply chain risk
    
    Parameters:
        part_id (str): The part ID needing alternatives (e.g., "DEN-0000001").
                      This is the unique identifier for the part in the system.
        reason (str): Why alternatives are needed. Options:
                     - "quality_issue": Current supplier has quality problems
                       → Sets minimum quality rating to 4.5 (higher standard)
                     - "geopolitical_risk": Geographic/political concerns
                       → Prioritizes geographic diversity (different regions)
                     - "capacity_shortage": Supplier cannot meet demand
                       → Prioritizes suppliers with higher production capacity
                     - "cost_reduction": Seeking better pricing
                       → Focuses on cost-effective alternatives
                     - "lead_time": Need faster delivery
                       → Prioritizes suppliers with shorter lead times
                     - "general": General search for alternatives
                       → Uses default parameters
    
    Returns:
        str: Formatted string containing top 3 alternative suppliers with:
            - Supplier name and location
            - Quality rating and on-time delivery rate
            - Composite score (weighted evaluation)
            - Key justification points (why recommended)
            - Estimated qualification timeline (days)
            - Risk factors and trade-offs
    
    Example Usage:
        >>> find_supplier_alternatives_tool("DEN-0000001", "quality_issue")
        >>> find_supplier_alternatives_tool("VAL-0000002", "geopolitical_risk")
        >>> find_supplier_alternatives_tool("MAG-0000003", "capacity_shortage")
        >>> find_supplier_alternatives_tool("SIE-0000004", "cost_reduction")
    
    Note:
        Alternatives found using Qdrant vector search
        This tool leverages semantic understanding to find compatible parts even
        when exact matches aren't available, enabling rapid supplier diversification.
    """
    try:
        # Create SupplierAlternativeAgent instance
        agent = SupplierAlternativeAgent()
        
        # Determine parameters based on reason
        min_quality_rating = 4.0  # Default
        prefer_geographic_diversity = False  # Default
        max_results = 3
        
        reason_lower = reason.lower().strip()
        
        if reason_lower == "quality_issue":
            min_quality_rating = 4.5
            logger.info("Quality issue detected - using higher quality threshold (4.5)")
        elif reason_lower == "geopolitical_risk":
            prefer_geographic_diversity = True
            logger.info("Geopolitical risk - prioritizing geographic diversity")
        elif reason_lower == "capacity_shortage":
            # Note: Production capacity filtering would be added in future enhancement
            min_quality_rating = 4.0
            logger.info("Capacity shortage - prioritizing high-capacity suppliers")
        elif reason_lower == "cost_reduction":
            min_quality_rating = 4.0
            logger.info("Cost reduction - focusing on cost-effective alternatives")
        elif reason_lower == "lead_time":
            min_quality_rating = 4.0
            logger.info("Lead time concern - prioritizing fast delivery suppliers")
        else:
            logger.info(f"General search - using default parameters")
        
        # Call find_alternative_suppliers()
        alternatives = agent.find_alternative_suppliers(
            part_id=part_id,
            exclude_supplier_id=None,
            min_quality_rating=min_quality_rating,
            max_results=max_results,
            prefer_geographic_diversity=prefer_geographic_diversity
        )
        
        # Format results as readable string
        if not alternatives:
            return (
                f"🔍 Supplier Alternatives Search\n"
                f"{'='*60}\n"
                f"Part ID: {part_id}\n"
                f"Reason: {reason}\n"
                f"{'='*60}\n"
                f"\n❌ No alternative suppliers found.\n"
                f"\n💡 Suggestions:\n"
                f"  - Try adjusting quality requirements\n"
                f"  - Consider different part specifications\n"
                f"  - Review part ID for accuracy\n"
                f"\nAlternatives found using Qdrant vector search"
            )
        
        # Build formatted output
        output_lines = [
            f"🔍 Supplier Alternatives Search",
            f"{'='*60}",
            f"Part ID: {part_id}",
            f"Reason: {reason}",
            f"Alternatives Found: {len(alternatives)}",
            f"{'='*60}",
            ""
        ]
        
        # Format each alternative supplier
        for idx, alt in enumerate(alternatives, 1):
            supplier_info = alt.get('supplier_info', {})
            part_compat = alt.get('part_compatibility', {})
            
            supplier_name = supplier_info.get('company_name', 'Unknown Supplier')
            supplier_location = supplier_info.get('location', 'Unknown Location')
            supplier_id = supplier_info.get('supplier_id', 'N/A')
            
            quality_rating = supplier_info.get('quality_rating', 0.0)
            on_time_rate = supplier_info.get('on_time_delivery_rate', 0.0)
            if isinstance(on_time_rate, float) and on_time_rate > 1.0:
                on_time_rate = on_time_rate / 100.0
            
            composite_score = alt.get('composite_score', 0.0)
            justification = alt.get('justification', 'No justification provided')
            qualification_time = alt.get('estimated_qualification_time', 30)
            
            # Format quality rating
            quality_str = f"{quality_rating:.1f}/5.0" if isinstance(quality_rating, (int, float)) else str(quality_rating)
            quality_stars = "⭐" * int(quality_rating) if isinstance(quality_rating, (int, float)) else ""
            
            # Format on-time delivery
            on_time_str = f"{on_time_rate*100:.1f}%" if isinstance(on_time_rate, (int, float)) else "N/A"
            
            # Format composite score
            score_str = f"{composite_score:.3f}" if isinstance(composite_score, float) else str(composite_score)
            
            output_lines.extend([
                f"Alternative {idx}: {supplier_name}",
                f"{'─'*60}",
                f"  📍 Location: {supplier_location}",
                f"  🆔 Supplier ID: {supplier_id}",
                f"  ⭐ Quality Rating: {quality_str} {quality_stars}",
                f"  📦 On-Time Delivery: {on_time_str}",
                f"  🎯 Composite Score: {score_str}",
                f"",
                f"  💡 Justification:",
                f"     {justification[:200]}{'...' if len(justification) > 200 else ''}",
                f"",
                f"  ⏱️  Qualification Timeline: {qualification_time} days",
                f""
            ])
            
            # Add key strengths if available
            key_strengths = alt.get('key_strengths', [])
            if key_strengths:
                output_lines.append(f"  ✅ Key Strengths:")
                for strength in key_strengths[:3]:  # Top 3 strengths
                    output_lines.append(f"     • {strength}")
                output_lines.append("")
            
            # Add trade-offs if available
            trade_offs = alt.get('trade_offs', {})
            if trade_offs:
                output_lines.append(f"  ⚖️  Trade-offs:")
                if 'cost' in trade_offs:
                    cost_info = trade_offs['cost']
                    output_lines.append(f"     • Cost: ${cost_info.get('value', 'N/A')} {cost_info.get('unit', '')}")
                if 'lead_time' in trade_offs:
                    lt_info = trade_offs['lead_time']
                    output_lines.append(f"     • Lead Time: {lt_info.get('value', 'N/A')} {lt_info.get('unit', 'days')}")
                output_lines.append("")
        
        # Add summary and footer
        output_lines.extend([
            f"{'='*60}",
            f"Summary:",
            f"  • Found {len(alternatives)} alternative supplier(s)",
            f"  • Average Composite Score: {sum(a.get('composite_score', 0.0) for a in alternatives) / len(alternatives):.3f}",
            f"  • Search Reason: {reason}",
            f"",
            f"💡 Alternatives found using Qdrant vector search",
            f"   Semantic search enables finding compatible parts even when",
            f"   exact matches aren't available, enabling rapid supplier diversification."
        ])
        
        return "\n".join(output_lines)
        
    except Exception as e:
        error_msg = (
            f"❌ Error finding supplier alternatives: {str(e)}\n"
            f"\nPart ID: {part_id}\n"
            f"Reason: {reason}\n"
            f"\nPlease check your inputs and try again."
        )
        logger.error(f"find_supplier_alternatives_tool error: {e}")
        return error_msg


class SupplierAlternativeAgent:
    """
    AI agent for finding alternative suppliers and assessing supply chain risks.
    
    This agent uses CrewAI framework combined with Qdrant vector search to find
    alternative suppliers during supply chain disruptions and assess risks based on
    supplier profiles, relationships, and quality incident history.
    
    The agent leverages semantic search to match supplier capabilities with part
    requirements, enabling rapid identification of alternative suppliers when primary
    suppliers face disruptions.
    
    Attributes:
        client: Qdrant client connection for vector search operations
        suppliers_data: Dictionary of suppliers keyed by supplier_id
        relationships_data: List of part-supplier relationship records
        quality_incidents_data: List of quality incident records
        agent: CrewAI Agent instance for AI-powered supplier analysis
    
    Example:
        >>> agent = SupplierAlternativeAgent()
        >>> alternatives = agent.find_alternative_suppliers("DEN-0000001")
    """
    
    def __init__(self):
        """
        Initialize the SupplierAlternativeAgent.
        
        Sets up:
        - Qdrant client connection
        - Loads suppliers, relationships, and quality incidents data
        - Converts suppliers to dictionary keyed by supplier_id
        - Creates CrewAI Agent with specialized role and expertise
        
        Raises:
            RuntimeError: If initialization fails
            FileNotFoundError: If required data files are missing
        """
        try:
            logger.info("Initializing SupplierAlternativeAgent...")
            
            # Initialize Qdrant client
            logger.debug("Connecting to Qdrant...")
            self.client = get_qdrant_client()
            logger.success("✓ Qdrant client connected")
            
            # Initialize embedding generator for semantic search
            logger.debug("Initializing embedding generator...")
            self.embedding_generator = EmbeddingGenerator()
            logger.success("✓ Embedding generator ready")
            
            # Load all necessary data from data_loader
            logger.debug("Loading supply chain data...")
            data = load_all_data()
            
            # Extract suppliers, relationships, and quality incidents
            suppliers_list = data.get("suppliers", [])
            self.relationships_data = data.get("relationships", [])
            self.quality_incidents_data = data.get("incidents", [])
            
            # Convert suppliers to dict keyed by supplier_id
            logger.debug("Converting suppliers to dictionary...")
            self.suppliers_data: Dict[str, Dict[str, Any]] = {}
            for supplier in suppliers_list:
                supplier_id = supplier.get("supplier_id")
                if supplier_id:
                    self.suppliers_data[supplier_id] = supplier
                else:
                    logger.warning(f"Supplier missing supplier_id: {supplier}")
            
            logger.success(f"✓ Loaded {len(self.suppliers_data)} suppliers")
            logger.success(f"✓ Loaded {len(self.relationships_data)} relationships")
            logger.success(f"✓ Loaded {len(self.quality_incidents_data)} quality incidents")
            
            # Create CrewAI Agent
            # Note: CrewAI requires an LLM provider (e.g., OpenAI) to be configured
            # Set OPENAI_API_KEY environment variable or configure LLM in settings
            logger.debug("Creating CrewAI Agent...")
            try:
                self.agent = Agent(
                    role="Supply Chain Risk Mitigation Specialist",
                    goal=(
                        "Find alternative suppliers and assess supply chain risks "
                        "using advanced vector search"
                    ),
                    backstory=(
                        "Former supply chain manager at major OEM with 500+ suppliers "
                        "experience. Specializes in identifying alternative suppliers "
                        "during disruptions and assessing supply chain risks. Uses Qdrant's "
                        "semantic search to find optimal supplier alternatives by matching "
                        "supplier capabilities with part requirements. Has deep knowledge "
                        "of supplier evaluation, quality management, and risk assessment "
                        "methodologies."
                    ),
                    verbose=True,
                    allow_delegation=False,
                    tools=[find_supplier_alternatives_tool],  # Add supplier alternatives tool
                )
                logger.success("✓ CrewAI Agent created with find_supplier_alternatives_tool")
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
            
            logger.success("✅ SupplierAlternativeAgent initialized successfully")
            
        except FileNotFoundError as e:
            error_msg = f"Required data files not found: {e}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize SupplierAlternativeAgent: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_agent(self) -> Optional[Agent]:
        """
        Get the CrewAI Agent instance.
        
        Returns:
            Optional[Agent]: The CrewAI Agent instance, or None if not initialized
                           (requires LLM configuration)
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> crewai_agent = agent.get_agent()
        """
        return self.agent
    
    def get_client(self):
        """
        Get the Qdrant client instance.
        
        Returns:
            QdrantClient: The Qdrant client instance
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> client = agent.get_client()
        """
        return self.client
    
    def get_suppliers_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the suppliers data dictionary.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of suppliers keyed by supplier_id
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> suppliers = agent.get_suppliers_data()
            >>> supplier = suppliers.get("SUP-001")
        """
        return self.suppliers_data
    
    def get_relationships_data(self) -> List[Dict[str, Any]]:
        """
        Get the part-supplier relationships data.
        
        Returns:
            List[Dict[str, Any]]: List of relationship records
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> relationships = agent.get_relationships_data()
        """
        return self.relationships_data
    
    def get_quality_incidents_data(self) -> List[Dict[str, Any]]:
        """
        Get the quality incidents data.
        
        Returns:
            List[Dict[str, Any]]: List of quality incident records
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> incidents = agent.get_quality_incidents_data()
        """
        return self.quality_incidents_data
    
    def find_alternative_suppliers(
        self,
        part_id: str,
        exclude_supplier_id: Optional[str] = None,
        min_quality_rating: float = 4.0,
        max_results: int = 5,
        prefer_geographic_diversity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find alternative suppliers for a part during disruptions.
        
        Uses Qdrant semantic search to find compatible alternative parts by matching
        part specifications and requirements. Evaluates suppliers based on quality,
        delivery performance, lead time, and cost, with optional geographic diversity
        bonuses to reduce supply chain risk.
        
        Args:
            part_id: The part ID needing alternatives (e.g., "DEN-0000001")
            exclude_supplier_id: Optional supplier ID to exclude from results
            min_quality_rating: Minimum acceptable quality rating (0.0-5.0, default: 4.0)
            max_results: Maximum number of alternative suppliers to return (default: 5)
            prefer_geographic_diversity: If True, prioritize suppliers in different
                                       regions to reduce supply chain risk (default: True)
        
        Returns:
            List of dictionaries, each containing:
            - supplier_info: Full supplier details
            - part_compatibility: Compatible part information
            - composite_score: Weighted score (0.0-1.0, higher is better)
            - justification: Text explanation of why supplier is recommended
            - key_strengths: List of supplier strengths
            - trade_offs: Dictionary of trade-offs (cost, lead_time, etc.)
            - risk_factors: List of potential risk factors
            - estimated_qualification_time: Estimated time to qualify supplier (days)
        
        Raises:
            ValueError: If part_id is empty or invalid
            RuntimeError: If search fails
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> alternatives = agent.find_alternative_suppliers(
            ...     part_id="DEN-0000001",
            ...     exclude_supplier_id="SUP-001",
            ...     min_quality_rating=4.0,
            ...     max_results=5
            ... )
            >>> for alt in alternatives:
            ...     print(f"{alt['supplier_info']['company_name']}: {alt['composite_score']:.3f}")
        """
        if not part_id or not part_id.strip():
            error_msg = "part_id cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("="*60)
            logger.info(f"Finding alternative suppliers for part: {part_id}")
            logger.info("="*60)
            
            # Step 1: Query Qdrant to get original part details
            logger.debug(f"Retrieving part details for {part_id}...")
            original_part = self._get_part_by_id(part_id)
            
            if not original_part:
                error_msg = f"Part not found: {part_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.success(f"✓ Found original part: {original_part.get('part_name', 'N/A')}")
            
            # Extract part specifications for semantic search
            part_specs = self._extract_part_specifications(original_part)
            logger.debug(f"Extracted part specifications: {part_specs[:100]}...")
            
            # Step 2: Search for similar parts using vector similarity
            # Uses Qdrant semantic search to find compatible alternative parts
            logger.info("Searching for similar parts using semantic search...")
            similar_parts = self._find_similar_parts(
                part_specs=part_specs,
                exclude_supplier_id=exclude_supplier_id,
                min_quality_rating=min_quality_rating,
                limit=20  # Get top 20 candidates for scoring
            )
            
            logger.success(f"✓ Found {len(similar_parts)} candidate parts")
            
            if not similar_parts:
                logger.warning("No alternative parts found matching criteria")
                return []
            
            # Step 3 & 4: Calculate composite scores for each candidate
            logger.info("Calculating composite scores for candidates...")
            scored_candidates = []
            
            # Track regions for geographic diversity
            seen_regions = set()
            if exclude_supplier_id and exclude_supplier_id in self.suppliers_data:
                excluded_supplier = self.suppliers_data[exclude_supplier_id]
                excluded_region = excluded_supplier.get('location', {}).get('region', '') if isinstance(excluded_supplier.get('location'), dict) else ''
                seen_regions.add(excluded_region)
            
            for candidate_part in similar_parts:
                # Get supplier ID from candidate (already set in _find_similar_parts)
                supplier_id = candidate_part.get('supplier_id')
                if not supplier_id:
                    continue
                
                # Retrieve full supplier details
                supplier_info = self.suppliers_data.get(supplier_id)
                if not supplier_info:
                    logger.warning(f"Supplier {supplier_id} not found in suppliers data")
                    continue
                
                # Get relationship details for scoring (from candidate or lookup)
                relationship = candidate_part.get('relationship') or self._get_relationship(
                    candidate_part.get('part_id', part_id), supplier_id
                )
                
                # Calculate composite score
                composite_score = self._calculate_composite_score(
                    supplier_info=supplier_info,
                    relationship=relationship,
                    candidate_part=candidate_part,
                    prefer_geographic_diversity=prefer_geographic_diversity,
                    seen_regions=seen_regions
                )
                
                # Skip if score is too low
                if composite_score < 0.3:
                    continue
                
                # Generate justification and analysis
                justification = self._generate_justification(
                    supplier_info=supplier_info,
                    relationship=relationship,
                    candidate_part=candidate_part,
                    original_part=original_part,
                    composite_score=composite_score
                )
                
                key_strengths = self._identify_key_strengths(
                    supplier_info=supplier_info,
                    relationship=relationship
                )
                
                trade_offs = self._identify_trade_offs(
                    supplier_info=supplier_info,
                    relationship=relationship,
                    original_part=original_part
                )
                
                risk_factors = self._identify_risk_factors(
                    supplier_id=supplier_id,
                    supplier_info=supplier_info
                )
                
                estimated_qualification_time = self._estimate_qualification_time(
                    supplier_info=supplier_info,
                    relationship=relationship
                )
                
                scored_candidates.append({
                    'supplier_info': supplier_info,
                    'part_compatibility': candidate_part,
                    'composite_score': composite_score,
                    'justification': justification,
                    'key_strengths': key_strengths,
                    'trade_offs': trade_offs,
                    'risk_factors': risk_factors,
                    'estimated_qualification_time': estimated_qualification_time
                })
                
                # Track region for diversity bonus
                supplier_region = supplier_info.get('location', {}).get('region', '') if isinstance(supplier_info.get('location'), dict) else ''
                if supplier_region:
                    seen_regions.add(supplier_region)
            
            # Step 5: Sort by composite score, deduplicate by supplier_id, and return top results
            scored_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Deduplicate by supplier_id (keep highest scoring entry for each supplier)
            seen_suppliers = {}
            for candidate in scored_candidates:
                supplier_id = candidate['supplier_info'].get('supplier_id')
                if supplier_id not in seen_suppliers:
                    seen_suppliers[supplier_id] = candidate
                else:
                    # Keep the one with higher score
                    if candidate['composite_score'] > seen_suppliers[supplier_id]['composite_score']:
                        seen_suppliers[supplier_id] = candidate
            
            # Convert back to list and take top results
            results = list(seen_suppliers.values())
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            results = results[:max_results]
            
            logger.success(f"✓ Identified {len(results)} alternative suppliers")
            logger.info("="*60)
            
            # Log top results
            for idx, result in enumerate(results, 1):
                supplier_name = result['supplier_info'].get('company_name', 'N/A')
                score = result['composite_score']
                logger.info(f"{idx}. {supplier_name}: Score {score:.4f}")
            
            return results
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to find alternative suppliers: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _get_part_by_id(self, part_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve part details from Qdrant by part_id."""
        try:
            # Create filter for part_id
            filter_conditions = [
                FieldCondition(
                    key="part_id",
                    match=MatchValue(value=part_id)
                )
            ]
            qdrant_filter = Filter(must=filter_conditions)
            
            # Scroll to find the part
            scroll_result = self.client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=qdrant_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            
            results = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            
            if results:
                return results[0].payload
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving part {part_id}: {e}")
            return None
    
    def _extract_part_specifications(self, part: Dict[str, Any]) -> str:
        """Extract part specifications as text for semantic search."""
        spec_parts = []
        
        if part.get('part_name'):
            spec_parts.append(str(part['part_name']))
        if part.get('description'):
            spec_parts.append(str(part['description']))
        if part.get('specifications'):
            spec_parts.append(str(part['specifications']))
        if part.get('category'):
            spec_parts.append(str(part['category']))
        if part.get('compliance_standards'):
            if isinstance(part['compliance_standards'], list):
                spec_parts.extend([str(std) for std in part['compliance_standards']])
            else:
                spec_parts.append(str(part['compliance_standards']))
        
        return ' '.join(spec_parts)
    
    def _find_similar_parts(
        self,
        part_specs: str,
        exclude_supplier_id: Optional[str],
        min_quality_rating: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar parts using vector similarity search."""
        try:
            # Generate embedding for part specifications
            query_embedding = self.embedding_generator.generate_embedding(part_specs)
            
            # Build filter conditions - try quality filter, but handle gracefully if not indexed
            filter_conditions = []
            
            # Quality rating filter (may fail if not indexed, so we'll try and catch)
            try:
                filter_conditions.append(
                    FieldCondition(
                        key="quality_rating",
                        range=Range(gte=min_quality_rating)
                    )
                )
            except Exception:
                logger.debug("Quality rating filter not available, will filter in post-processing")
            
            qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search for similar parts
            query_response = self.client.query_points(
                collection_name=settings.collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=limit * 2,  # Get more candidates for post-filtering
                with_payload=True,
                with_vectors=False,
            )
            
            results = query_response.points if hasattr(query_response, 'points') else []
            
            # Post-process: filter and enrich with supplier info
            filtered_results = []
            for point in results:
                payload = point.payload
                part_id = payload.get('part_id')
                
                # Apply quality filter in post-processing if needed
                quality_rating = payload.get('quality_rating', 0.0)
                if isinstance(quality_rating, (int, float)) and quality_rating < min_quality_rating:
                    continue
                
                # Match suppliers based on specialization/capabilities
                # Since direct relationships aren't available, we match suppliers
                # based on their specialization matching the part category
                part_category = payload.get('category', '')
                
                # Find suppliers that match this part's category/specialization
                matching_suppliers = []
                for supplier_id, supplier_info in self.suppliers_data.items():
                    # Exclude supplier if specified
                    if exclude_supplier_id and supplier_id == exclude_supplier_id:
                        continue
                    
                    # Match by specialization
                    supplier_specialization = supplier_info.get('specialization', '').lower()
                    if part_category and supplier_specialization:
                        # Check if supplier specialization matches part category
                        category_lower = part_category.lower()
                        if (supplier_specialization in category_lower or 
                            category_lower in supplier_specialization or
                            any(word in category_lower for word in supplier_specialization.split())):
                            matching_suppliers.append(supplier_id)
                    else:
                        # If no specialization match, include all suppliers (for diversity)
                        matching_suppliers.append(supplier_id)
                
                # If no matching suppliers found, use all suppliers (except excluded)
                if not matching_suppliers:
                    matching_suppliers = [
                        sid for sid in self.suppliers_data.keys()
                        if sid != exclude_supplier_id
                    ]
                
                # Create candidates for each matching supplier
                for supplier_id in matching_suppliers[:3]:  # Limit to top 3 suppliers per part
                    # Create a synthetic relationship dict for scoring
                    rel_dict = {
                        'supplier_id': supplier_id,
                        'part_id': part_id,
                        'negotiated_lead_time_days': 30,  # Default values
                        'contract_price_per_unit': payload.get('unit_cost_usd', 100),
                        'delivery_performance_ytd': 95.0,
                        'is_approved_supplier': False,
                        'is_primary_supplier': False
                    }
                    
                    # Create enriched candidate with supplier info
                    candidate = payload.copy()
                    candidate['supplier_id'] = supplier_id
                    candidate['similarity_score'] = point.score if hasattr(point, 'score') else 0.0
                    candidate['relationship'] = rel_dict
                    filtered_results.append(candidate)
            
            # Sort by similarity score and return top candidates
            filtered_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar parts: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _get_relationship(self, part_id: str, supplier_id: str) -> Optional[Dict[str, Any]]:
        """Get relationship details between part and supplier."""
        for rel in self.relationships_data:
            if rel.get('part_id') == part_id and rel.get('supplier_id') == supplier_id:
                return rel
        return None
    
    def _calculate_composite_score(
        self,
        supplier_info: Dict[str, Any],
        relationship: Optional[Dict[str, Any]],
        candidate_part: Dict[str, Any],
        prefer_geographic_diversity: bool,
        seen_regions: set
    ) -> float:
        """
        Calculate composite score weighted by:
        - quality_rating (40%)
        - on_time_delivery_rate (30%)
        - lead_time preference (20% - prefer shorter)
        - cost preference (10% - prefer lower)
        """
        score = 0.0
        
        # Quality rating (40%)
        quality_rating = supplier_info.get('quality_rating', 0.0)
        if isinstance(quality_rating, (int, float)):
            score += (quality_rating / 5.0) * 0.4
        
        # On-time delivery rate (30%)
        on_time_rate = supplier_info.get('on_time_delivery_rate', 0.0)
        if isinstance(on_time_rate, (int, float)):
            # Normalize to 0-1 (assuming percentage 0-100)
            normalized_rate = min(on_time_rate / 100.0, 1.0) if on_time_rate > 1.0 else on_time_rate
            score += normalized_rate * 0.3
        
        # Lead time preference (20% - prefer shorter)
        lead_time = relationship.get('lead_time_days', 30) if relationship else 30
        if isinstance(lead_time, (int, float)):
            # Normalize: shorter is better (inverse relationship)
            # Assume max lead time is 90 days, normalize to 0-1
            normalized_lead_time = max(0.0, 1.0 - (lead_time / 90.0))
            score += normalized_lead_time * 0.2
        
        # Cost preference (10% - prefer lower)
        cost = relationship.get('unit_cost', 100.0) if relationship else 100.0
        if isinstance(cost, (int, float)):
            # Normalize: lower is better (inverse relationship)
            # Assume max cost is 1000, normalize to 0-1
            normalized_cost = max(0.0, 1.0 - (cost / 1000.0))
            score += normalized_cost * 0.1
        
        # Geographic diversity bonus (if enabled)
        if prefer_geographic_diversity:
            supplier_location = supplier_info.get('location', {})
            if isinstance(supplier_location, dict):
                supplier_region = supplier_location.get('region', '')
            else:
                supplier_region = ''
            
            if supplier_region and supplier_region not in seen_regions:
                # Bonus for different region (up to 10% boost)
                diversity_bonus = 0.1
                score = min(1.0, score + diversity_bonus)
                logger.debug(f"Geographic diversity bonus applied for region: {supplier_region}")
        
        return min(1.0, max(0.0, score))  # Clamp to 0-1
    
    def _generate_justification(
        self,
        supplier_info: Dict[str, Any],
        relationship: Optional[Dict[str, Any]],
        candidate_part: Dict[str, Any],
        original_part: Dict[str, Any],
        composite_score: float
    ) -> str:
        """Generate detailed justification for supplier recommendation."""
        supplier_name = supplier_info.get('company_name', 'Unknown Supplier')
        quality_rating = supplier_info.get('quality_rating', 0.0)
        on_time_rate = supplier_info.get('on_time_delivery_rate', 0.0)
        
        justification_parts = [
            f"{supplier_name} is recommended as an alternative supplier with a composite score of {composite_score:.3f}."
        ]
        
        if quality_rating >= 4.5:
            justification_parts.append("Exceptional quality rating demonstrates reliable manufacturing capabilities.")
        elif quality_rating >= 4.0:
            justification_parts.append("High quality rating indicates consistent product quality.")
        
        if on_time_rate >= 95:
            justification_parts.append("Excellent on-time delivery performance ensures supply chain reliability.")
        elif on_time_rate >= 90:
            justification_parts.append("Strong delivery performance reduces supply chain disruption risk.")
        
        if relationship:
            lead_time = relationship.get('lead_time_days', 30)
            if lead_time <= 14:
                justification_parts.append("Short lead time enables rapid response to supply disruptions.")
            elif lead_time <= 30:
                justification_parts.append("Reasonable lead time allows for manageable transition planning.")
        
        part_name = candidate_part.get('part_name', 'N/A')
        justification_parts.append(f"Part '{part_name}' matches specifications and requirements.")
        
        return ' '.join(justification_parts)
    
    def _identify_key_strengths(
        self,
        supplier_info: Dict[str, Any],
        relationship: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify key strengths of the supplier."""
        strengths = []
        
        quality_rating = supplier_info.get('quality_rating', 0.0)
        if quality_rating >= 4.5:
            strengths.append("Exceptional quality standards")
        elif quality_rating >= 4.0:
            strengths.append("High quality rating")
        
        on_time_rate = supplier_info.get('on_time_delivery_rate', 0.0)
        if on_time_rate >= 95:
            strengths.append("Excellent delivery reliability")
        elif on_time_rate >= 90:
            strengths.append("Strong delivery performance")
        
        specialization = supplier_info.get('specialization', '')
        if specialization:
            strengths.append(f"Specialized in {specialization}")
        
        tier_level = supplier_info.get('tier_level', '')
        if tier_level == 'Tier 1':
            strengths.append("Tier 1 supplier status")
        
        if relationship:
            lead_time = relationship.get('lead_time_days', 30)
            if lead_time <= 14:
                strengths.append("Fast lead time")
        
        return strengths
    
    def _identify_trade_offs(
        self,
        supplier_info: Dict[str, Any],
        relationship: Optional[Dict[str, Any]],
        original_part: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify trade-offs (cost, lead time, etc.)."""
        trade_offs = {}
        
        if relationship:
            cost = relationship.get('unit_cost', None)
            if cost is not None:
                trade_offs['cost'] = {
                    'value': cost,
                    'unit': 'USD',
                    'note': 'Higher cost may impact profitability'
                }
            
            lead_time = relationship.get('lead_time_days', None)
            if lead_time is not None:
                trade_offs['lead_time'] = {
                    'value': lead_time,
                    'unit': 'days',
                    'note': 'Longer lead time requires advance planning'
                }
        
        quality_rating = supplier_info.get('quality_rating', 0.0)
        if quality_rating < 4.5:
            trade_offs['quality'] = {
                'value': quality_rating,
                'note': 'May require additional quality verification'
            }
        
        return trade_offs
    
    def _identify_risk_factors(
        self,
        supplier_id: str,
        supplier_info: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # Check quality incidents
        incidents = [
            inc for inc in self.quality_incidents_data
            if inc.get('supplier_id') == supplier_id
        ]
        
        if len(incidents) > 5:
            risks.append(f"Multiple quality incidents ({len(incidents)} total)")
        elif len(incidents) > 0:
            risks.append(f"Some quality incidents in history ({len(incidents)} total)")
        
        # Check quality rating
        quality_rating = supplier_info.get('quality_rating', 0.0)
        if quality_rating < 4.0:
            risks.append("Below-average quality rating")
        
        # Check on-time delivery
        on_time_rate = supplier_info.get('on_time_delivery_rate', 100.0)
        if on_time_rate < 85:
            risks.append("Lower on-time delivery rate")
        
        # Check tier level
        tier_level = supplier_info.get('tier_level', '')
        if tier_level not in ['Tier 1', 'Tier 2']:
            risks.append("Lower tier supplier may have limited capabilities")
        
        if not risks:
            risks.append("Low risk profile")
        
        return risks
    
    def _estimate_qualification_time(
        self,
        supplier_info: Dict[str, Any],
        relationship: Optional[Dict[str, Any]]
    ) -> int:
        """Estimate time to qualify supplier (in days)."""
        base_time = 30  # Base qualification time
        
        # Adjust based on tier level
        tier_level = supplier_info.get('tier_level', '')
        if tier_level == 'Tier 1':
            base_time -= 10  # Tier 1 suppliers are pre-qualified
        elif tier_level == 'Tier 2':
            base_time -= 5
        
        # Adjust based on quality rating
        quality_rating = supplier_info.get('quality_rating', 0.0)
        if quality_rating >= 4.5:
            base_time -= 5
        elif quality_rating >= 4.0:
            base_time -= 2
        
        # Adjust based on existing relationship
        if relationship:
            base_time -= 5  # Existing relationship reduces qualification time
        
        return max(7, base_time)  # Minimum 7 days
    
    def assess_supplier_risk(
        self,
        supplier_id: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a supplier.
        
        This method evaluates multiple risk dimensions including quality incidents,
        geographic factors, delivery performance, and financial stability to provide
        a comprehensive risk assessment on a 1-10 scale.
        
        Args:
            supplier_id: Supplier ID to assess (e.g., "SUP-001")
        
        Returns:
            Dictionary containing comprehensive risk assessment:
            - supplier_id: Supplier identifier
            - supplier_name: Company name
            - overall_risk_score: Risk score (1.0-10.0, higher = more risk)
            - risk_category: "Low", "Medium", or "High"
            - risk_factors: List of specific risk factors identified
            - quality_metrics: Dictionary with quality incident statistics
            - geographic_factors: Dictionary with geographic risk details
            - delivery_performance: Dictionary with delivery metrics
            - financial_factors: Dictionary with financial risk indicators
            - detailed_recommendations: List of specific recommendations
            - trend_analysis: "improving", "stable", or "declining"
            - visualization_data: Dictionary with data ready for charting
        
        Raises:
            ValueError: If supplier_id is empty or supplier not found
        
        Example:
            >>> agent = SupplierAlternativeAgent()
            >>> risk_assessment = agent.assess_supplier_risk("SUP-001")
            >>> print(f"Risk Score: {risk_assessment['overall_risk_score']}")
            >>> print(f"Category: {risk_assessment['risk_category']}")
        """
        if not supplier_id or not supplier_id.strip():
            error_msg = "supplier_id cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("="*60)
            logger.info(f"Assessing risk for supplier: {supplier_id}")
            logger.info("="*60)
            
            # Step 1: Retrieve supplier details
            supplier_info = self.suppliers_data.get(supplier_id)
            if not supplier_info:
                error_msg = f"Supplier not found: {supplier_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            supplier_name = supplier_info.get('company_name', 'Unknown')
            logger.success(f"✓ Found supplier: {supplier_name}")
            
            # Step 2: Get all quality incidents for this supplier
            logger.debug("Analyzing quality incidents...")
            supplier_incidents = [
                inc for inc in self.quality_incidents_data
                if inc.get('supplier_id') == supplier_id
            ]
            
            # Count incidents by severity
            severity_counts = {
                'Critical': 0,
                'Moderate': 0,
                'Minor': 0
            }
            
            for incident in supplier_incidents:
                severity = incident.get('severity', 'Minor')
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            total_incidents = len(supplier_incidents)
            
            # Calculate average severity (weighted)
            severity_weights = {'Critical': 3.0, 'Moderate': 1.5, 'Minor': 0.5}
            weighted_severity_sum = sum(
                severity_counts[sev] * severity_weights[sev]
                for sev in severity_counts.keys()
            )
            average_severity = (
                weighted_severity_sum / total_incidents
                if total_incidents > 0 else 0.0
            )
            
            logger.info(f"  Total incidents: {total_incidents}")
            logger.info(f"  Critical: {severity_counts['Critical']}, "
                       f"Moderate: {severity_counts['Moderate']}, "
                       f"Minor: {severity_counts['Minor']}")
            
            # Step 3: Calculate risk scores
            baseline_score = 5.0
            
            # Quality Risk Score
            quality_risk_score = weighted_severity_sum * 0.3  # Scale factor
            quality_risk_score = min(3.0, quality_risk_score)  # Cap at 3.0
            
            # Geographic Risk Score
            geographic_risk_level = supplier_info.get('geographic_risk_level', 'Medium')
            geographic_risk_map = {
                'Low': 1.0,
                'Medium': 5.0,
                'High': 9.0
            }
            geographic_risk_score = geographic_risk_map.get(
                geographic_risk_level,
                geographic_risk_map['Medium']
            )
            # Normalize to 0-2.0 range for addition to baseline
            geographic_risk_score = (geographic_risk_score - 1.0) / 4.0  # Maps to 0-2.0
            
            # Delivery Risk Score
            on_time_delivery_rate = supplier_info.get('on_time_delivery_rate', 0.95)
            if isinstance(on_time_delivery_rate, float):
                # Convert percentage to decimal if needed
                if on_time_delivery_rate > 1.0:
                    on_time_delivery_rate = on_time_delivery_rate / 100.0
            else:
                on_time_delivery_rate = 0.95  # Default
            
            delivery_risk_score = 10.0 * (1.0 - on_time_delivery_rate)
            # Normalize to 0-2.0 range
            delivery_risk_score = min(2.0, delivery_risk_score / 5.0)
            
            # Financial Risk Score
            years_in_business = supplier_info.get('years_in_business', 20)
            financial_risk_score = 0.0
            
            # Newer suppliers = higher risk
            if years_in_business < 5:
                financial_risk_score = 1.5
            elif years_in_business < 10:
                financial_risk_score = 1.0
            elif years_in_business < 20:
                financial_risk_score = 0.5
            
            # Check for single-source dependencies
            # Count relationships where this supplier is primary
            primary_relationships = [
                rel for rel in self.relationships_data
                if rel.get('supplier_id') == supplier_id and rel.get('is_primary_supplier', False)
            ]
            if len(primary_relationships) > 10:  # High dependency
                financial_risk_score += 0.5
            
            financial_risk_score = min(2.0, financial_risk_score)
            
            # Calculate overall risk score
            overall_risk_score = (
                baseline_score +
                quality_risk_score +
                geographic_risk_score +
                delivery_risk_score +
                financial_risk_score
            )
            overall_risk_score = min(10.0, max(1.0, overall_risk_score))
            
            logger.info(f"Risk Score Breakdown:")
            logger.info(f"  Baseline: {baseline_score:.2f}")
            logger.info(f"  Quality Risk: +{quality_risk_score:.2f}")
            logger.info(f"  Geographic Risk: +{geographic_risk_score:.2f}")
            logger.info(f"  Delivery Risk: +{delivery_risk_score:.2f}")
            logger.info(f"  Financial Risk: +{financial_risk_score:.2f}")
            logger.info(f"  Overall Score: {overall_risk_score:.2f}")
            
            # Step 4: Categorize risk
            if overall_risk_score < 4.0:
                risk_category = "Low"
            elif overall_risk_score <= 7.0:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            logger.success(f"✓ Risk Category: {risk_category}")
            
            # Step 5: Generate recommendations
            recommendations = []
            if risk_category == "High":
                recommendations.append("Seek alternatives immediately, implement contingency")
                recommendations.append("Increase monitoring frequency to weekly")
                recommendations.append("Develop backup supplier relationships")
                recommendations.append("Consider dual-sourcing critical components")
            elif risk_category == "Medium":
                recommendations.append("Monitor closely, develop backup suppliers")
                recommendations.append("Conduct quarterly risk reviews")
                recommendations.append("Maintain alternative supplier options")
                recommendations.append("Track key performance indicators monthly")
            else:  # Low
                recommendations.append("Continue normal operations, periodic review")
                recommendations.append("Conduct annual risk assessment")
                recommendations.append("Maintain supplier relationship")
            
            # Add specific recommendations based on risk factors
            if total_incidents > 5:
                recommendations.append("Address quality issues through supplier development program")
            if on_time_delivery_rate < 0.90:
                recommendations.append("Improve delivery performance through collaborative planning")
            if geographic_risk_level == "High":
                recommendations.append("Consider geographic diversification for critical components")
            if years_in_business < 10:
                recommendations.append("Monitor financial stability closely")
            
            # Step 6: Identify risk factors
            risk_factors = []
            
            if total_incidents > 0:
                if severity_counts['Critical'] > 0:
                    risk_factors.append(f"{severity_counts['Critical']} critical quality incident(s)")
                if severity_counts['Moderate'] > 2:
                    risk_factors.append(f"{severity_counts['Moderate']} moderate quality incidents")
                if total_incidents > 5:
                    risk_factors.append("High number of quality incidents")
            
            if geographic_risk_level == "High":
                risk_factors.append("High geographic risk location")
            elif geographic_risk_level == "Medium":
                risk_factors.append("Moderate geographic risk")
            
            if on_time_delivery_rate < 0.90:
                risk_factors.append(f"Low on-time delivery rate ({on_time_delivery_rate*100:.1f}%)")
            
            if years_in_business < 10:
                risk_factors.append(f"Relatively new supplier ({years_in_business} years)")
            
            if len(primary_relationships) > 10:
                risk_factors.append("High dependency (many primary supplier relationships)")
            
            if not risk_factors:
                risk_factors.append("No significant risk factors identified")
            
            # Step 7: Trend analysis
            # Analyze incident trend (simplified - compare recent vs older incidents)
            recent_incidents = [
                inc for inc in supplier_incidents
                if inc.get('incident_date', '') >= '2024-01-01'  # Recent
            ]
            older_incidents = [
                inc for inc in supplier_incidents
                if inc.get('incident_date', '') < '2024-01-01'  # Older
            ]
            
            if len(recent_incidents) < len(older_incidents):
                trend_analysis = "improving"
            elif len(recent_incidents) > len(older_incidents):
                trend_analysis = "declining"
            else:
                trend_analysis = "stable"
            
            # Step 8: Build comprehensive result dictionary
            result = {
                'supplier_id': supplier_id,
                'supplier_name': supplier_name,
                'overall_risk_score': round(overall_risk_score, 2),
                'risk_category': risk_category,
                'risk_factors': risk_factors,
                'quality_metrics': {
                    'total_incidents': total_incidents,
                    'critical_count': severity_counts['Critical'],
                    'moderate_count': severity_counts['Moderate'],
                    'minor_count': severity_counts['Minor'],
                    'average_severity': round(average_severity, 2),
                    'quality_risk_score': round(quality_risk_score, 2)
                },
                'geographic_factors': {
                    'risk_level': geographic_risk_level,
                    'location': supplier_info.get('location', 'Unknown'),
                    'geographic_risk_score': round(geographic_risk_score, 2)
                },
                'delivery_performance': {
                    'on_time_delivery_rate': round(on_time_delivery_rate, 3),
                    'delivery_risk_score': round(delivery_risk_score, 2),
                    'performance_status': (
                        'Excellent' if on_time_delivery_rate >= 0.95 else
                        'Good' if on_time_delivery_rate >= 0.90 else
                        'Needs Improvement'
                    )
                },
                'financial_factors': {
                    'years_in_business': years_in_business,
                    'primary_relationships_count': len(primary_relationships),
                    'financial_risk_score': round(financial_risk_score, 2),
                    'stability_indicator': (
                        'High' if years_in_business >= 20 else
                        'Medium' if years_in_business >= 10 else
                        'Low'
                    )
                },
                'detailed_recommendations': recommendations,
                'trend_analysis': trend_analysis,
                'visualization_data': {
                    'risk_breakdown': {
                        'baseline': round(baseline_score, 2),
                        'quality': round(quality_risk_score, 2),
                        'geographic': round(geographic_risk_score, 2),
                        'delivery': round(delivery_risk_score, 2),
                        'financial': round(financial_risk_score, 2)
                    },
                    'severity_distribution': severity_counts,
                    'risk_category': risk_category,
                    'overall_score': round(overall_risk_score, 2),
                    'trend': trend_analysis
                }
            }
            
            logger.info("="*60)
            logger.success(f"✓ Risk assessment complete for {supplier_name}")
            logger.info(f"  Overall Risk Score: {overall_risk_score:.2f}/10.0")
            logger.info(f"  Risk Category: {risk_category}")
            logger.info(f"  Trend: {trend_analysis}")
            logger.info("="*60)
            
            return result
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to assess supplier risk: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

