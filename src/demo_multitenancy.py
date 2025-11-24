"""
Qdrant 1.16 Tiered Multitenancy Demonstration

This module demonstrates Qdrant 1.16's Tiered Multitenancy capabilities:
- Tenant-isolated searches (proving data isolation)
- Cross-tenant searches (unified architecture benefits)
- Tenant promotion scenarios (scaling from shared to dedicated shards)
- Hybrid workloads (combining tenant-specific and cross-tenant queries)

Key Concepts:
- Qdrant routes queries to correct tenants using oem_id filters
- Single collection is better than multiple collections (simplified operations)
- "Noisy neighbor" problem: high-volume tenants can impact others
- Solution: Promote high-volume tenants to dedicated shards seamlessly
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from loguru import logger

from src.agents.parts_discovery_agent import PartsDiscoveryAgent
from src.data_loader import load_all_data
from src.qdrant_client import get_qdrant_client
from src.config import settings


# Initialize rich console for beautiful output
console = Console()


def run_multitenancy_demo() -> Dict[str, Any]:
    """
    Comprehensive demonstration of Qdrant 1.16 Tiered Multitenancy capabilities.
    
    Demonstrates:
    1. Tenant-isolated searches (proving isolation)
    2. Cross-tenant searches (unified architecture)
    3. Tenant promotion scenarios (scaling concepts)
    4. Hybrid workloads (practical use cases)
    
    Returns:
        Dictionary containing all demo results and metrics
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Qdrant 1.16 Tiered Multitenancy Demonstration[/bold cyan]",
        border_style="cyan"
    ))
    console.print("\n")
    
    # Create results directory
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent and load data
    console.print("[yellow]Initializing PartsDiscoveryAgent and loading data...[/yellow]")
    try:
        agent = PartsDiscoveryAgent()
        data = load_all_data()
        oems_data = data.get('oems', [])
        
        # Create OEM lookup dictionary
        oems_dict = {oem['oem_id']: oem for oem in oems_data}
        
        console.print("[green]✓ Agent initialized successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize: {e}[/red]")
        return {"error": str(e)}
    
    demo_results = {
        "demo_timestamp": datetime.now().isoformat(),
        "demos": {}
    }
    
    # ========================================================================
    # DEMO 1: Tenant-Isolated Searches (Proving Isolation)
    # ========================================================================
    console.print(Panel(
        "[bold blue]DEMO 1: Tenant-Isolated Searches[/bold blue]\n"
        "[dim]Proving that each OEM's data is completely isolated[/dim]",
        border_style="blue",
        title="Tenant Isolation Verification"
    ))
    console.print("\n")
    
    demo1_results = []
    query = "battery components for electric vehicles"
    
    console.print(f"[cyan]Query:[/cyan] {query}\n")
    
    for oem_id in ['OEM-A', 'OEM-B', 'OEM-C']:
        oem_info = oems_dict.get(oem_id, {})
        company_name = oem_info.get('company_name', oem_id)
        market_segment = oem_info.get('market_segment', 'Unknown')
        
        console.print(f"[dim]Searching {oem_id} ({company_name})...[/dim]")
        
        try:
            # Execute tenant-isolated search
            results = agent.search_parts_semantic(
                query=query,
                filters={},
                oem_id=oem_id,
                limit=20
            )
            
            # Verify isolation - check all results have correct oem_id
            # Results are ScoredPoint objects with .payload and .score attributes
            isolated_results = []
            contamination_count = 0
            
            for result in results:
                payload = result.payload if hasattr(result, 'payload') else {}
                result_oem_id = payload.get('oem_id', '') if isinstance(payload, dict) else ''
                
                if result_oem_id == oem_id:
                    isolated_results.append(result)
                else:
                    contamination_count += 1
                    logger.warning(f"⚠️  Contamination detected: Expected {oem_id}, got {result_oem_id}")
            
            # Calculate metrics
            quality_ratings = []
            for r in isolated_results:
                payload = r.payload if hasattr(r, 'payload') else {}
                if isinstance(payload, dict) and payload.get('quality_rating'):
                    quality_ratings.append(payload['quality_rating'])
            avg_quality = sum(quality_ratings) / len(quality_ratings) if quality_ratings else 0
            
            # Get top categories
            categories = defaultdict(int)
            for result in isolated_results[:10]:
                payload = result.payload if hasattr(result, 'payload') else {}
                if isinstance(payload, dict):
                    cat = payload.get('category', 'Unknown')
                    categories[cat] += 1
            top_categories = ', '.join([f"{k}({v})" for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]])
            
            # Get top 3 parts
            top_parts = []
            for r in isolated_results[:3]:
                payload = r.payload if hasattr(r, 'payload') else {}
                if isinstance(payload, dict):
                    top_parts.append({
                        'name': payload.get('part_name', 'Unknown'),
                        'category': payload.get('category', 'Unknown'),
                        'quality': payload.get('quality_rating', 0)
                    })
            
            # Verify isolation
            isolation_verified = contamination_count == 0
            
            result_data = {
                "oem_id": oem_id,
                "company_name": company_name,
                "market_segment": market_segment,
                "results_count": len(isolated_results),
                "avg_quality_rating": round(avg_quality, 2),
                "top_categories": top_categories,
                "top_parts": top_parts,
                "isolation_verified": isolation_verified,
                "contamination_count": contamination_count
            }
            
            demo1_results.append(result_data)
            
            if isolation_verified:
                console.print(f"[green]✓ {oem_id}: {len(isolated_results)} results, isolation verified[/green]")
            else:
                console.print(f"[red]✗ {oem_id}: {contamination_count} contamination errors[/red]")
                
        except Exception as e:
            logger.error(f"Error searching {oem_id}: {e}")
            console.print(f"[red]✗ Error searching {oem_id}: {e}[/red]")
    
    # Display Demo 1 results table
    console.print("\n")
    table1 = Table(title="Tenant Isolation Results", box=box.ROUNDED, show_header=True)
    table1.add_column("OEM ID", style="cyan", no_wrap=True)
    table1.add_column("Company Name", style="magenta")
    table1.add_column("Results", justify="right", style="green")
    table1.add_column("Avg Quality", justify="right", style="yellow")
    table1.add_column("Top Categories", style="dim")
    table1.add_column("Market Segment", style="blue")
    table1.add_column("Isolation", justify="center")
    
    for result in demo1_results:
        isolation_status = "[green]✓[/green]" if result['isolation_verified'] else "[red]✗[/red]"
        table1.add_row(
            result['oem_id'],
            result['company_name'],
            str(result['results_count']),
            f"{result['avg_quality_rating']:.2f}",
            result['top_categories'][:40] + "..." if len(result['top_categories']) > 40 else result['top_categories'],
            result['market_segment'],
            isolation_status
        )
    
    console.print(table1)
    
    # Verification summary
    all_verified = all(r['isolation_verified'] for r in demo1_results)
    if all_verified:
        console.print("\n[bold green]✓ Tenant isolation verified - no data leakage[/bold green]")
        console.print("[dim]All results correctly filtered by oem_id[/dim]\n")
    else:
        console.print("\n[bold red]✗ Tenant isolation verification failed[/bold red]\n")
    
    demo_results["demos"]["demo1_tenant_isolation"] = {
        "query": query,
        "results": demo1_results,
        "isolation_verified": all_verified,
        "timestamp": datetime.now().isoformat()
    }
    
    # ========================================================================
    # DEMO 2: Cross-Tenant Search (Unified Architecture)
    # ========================================================================
    console.print("\n")
    console.print(Panel(
        "[bold green]DEMO 2: Cross-Tenant Search[/bold green]\n"
        "[dim]Demonstrating unified architecture - search across all OEMs[/dim]",
        border_style="green",
        title="Unified Search Architecture"
    ))
    console.print("\n")
    
    query2 = "high-performance brake systems with ASIL-D rating"
    console.print(f"[cyan]Query:[/cyan] {query2}\n")
    console.print("[dim]Searching across ALL tenants (no oem_id filter)...[/dim]\n")
    
    try:
        # Execute cross-tenant search (no oem_id filter)
        cross_tenant_results = agent.search_parts_semantic(
            query=query2,
            filters={},
            oem_id=None,  # No tenant filter = cross-tenant search
            limit=30
        )
        
        # Group results by OEM
        # Results are ScoredPoint objects with .payload and .score attributes
        results_by_oem = defaultdict(list)
        for result in cross_tenant_results:
            payload = result.payload if hasattr(result, 'payload') else {}
            if isinstance(payload, dict):
                oem_id = payload.get('oem_id', 'Unknown')
                results_by_oem[oem_id].append(result)
        
        # Analyze by OEM
        oem_analysis = []
        for oem_id, oem_results in results_by_oem.items():
            oem_info = oems_dict.get(oem_id, {})
            company_name = oem_info.get('company_name', oem_id)
            
            quality_ratings = []
            prices = []
            lead_times = []
            
            for r in oem_results:
                payload = r.payload if hasattr(r, 'payload') else {}
                if isinstance(payload, dict):
                    if payload.get('quality_rating'):
                        quality_ratings.append(payload['quality_rating'])
                    if payload.get('price_usd'):
                        prices.append(payload['price_usd'])
                    elif payload.get('unit_cost_usd'):  # Fallback to unit_cost_usd
                        prices.append(payload['unit_cost_usd'])
                    if payload.get('lead_time_days'):
                        lead_times.append(payload['lead_time_days'])
            
            avg_quality = sum(quality_ratings) / len(quality_ratings) if quality_ratings else 0
            avg_price = sum(prices) / len(prices) if prices else 0
            avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
            
            oem_analysis.append({
                "oem_id": oem_id,
                "company_name": company_name,
                "results_count": len(oem_results),
                "avg_quality": round(avg_quality, 2),
                "avg_price": round(avg_price, 2),
                "avg_lead_time": round(avg_lead_time, 1)
            })
        
        # Sort by quality rating
        oem_analysis.sort(key=lambda x: x['avg_quality'], reverse=True)
        
        # Display comparative analysis
        console.print("[bold]Comparative Analysis:[/bold]\n")
        
        analysis_table = Table(title="Cross-Tenant Brake Systems Comparison", box=box.ROUNDED)
        analysis_table.add_column("OEM", style="cyan")
        analysis_table.add_column("Options", justify="right", style="green")
        analysis_table.add_column("Avg Quality", justify="right", style="yellow")
        analysis_table.add_column("Avg Price", justify="right", style="magenta")
        analysis_table.add_column("Avg Lead Time", justify="right", style="blue")
        
        for analysis in oem_analysis:
            analysis_table.add_row(
                analysis['company_name'],
                str(analysis['results_count']),
                f"{analysis['avg_quality']:.2f}",
                f"${analysis['avg_price']:,.0f}" if analysis['avg_price'] > 0 else "N/A",
                f"{analysis['avg_lead_time']:.0f} days" if analysis['avg_lead_time'] > 0 else "N/A"
            )
        
        console.print(analysis_table)
        
        # Find best OEM by quality
        best_quality_oem = oem_analysis[0] if oem_analysis else None
        most_options_oem = max(oem_analysis, key=lambda x: x['results_count']) if oem_analysis else None
        
        console.print("\n[bold]Key Insights:[/bold]")
        if best_quality_oem:
            console.print(f"  • [green]Best Quality:[/green] {best_quality_oem['company_name']} "
                         f"(Quality: {best_quality_oem['avg_quality']:.2f})")
        if most_options_oem:
            console.print(f"  • [cyan]Most Options:[/cyan] {most_options_oem['company_name']} "
                         f"({most_options_oem['results_count']} parts)")
        
        # Display detailed results table
        console.print("\n")
        details_table = Table(title="Cross-Tenant Search Results", box=box.ROUNDED)
        details_table.add_column("Part Name", style="cyan")
        details_table.add_column("OEM", style="magenta")
        details_table.add_column("Quality", justify="right", style="yellow")
        details_table.add_column("Supplier", style="dim")
        details_table.add_column("Price", justify="right", style="green")
        details_table.add_column("Lead Time", justify="right", style="blue")
        
        # Show top 10 results
        for result in cross_tenant_results[:10]:
            payload = result.payload if hasattr(result, 'payload') else {}
            if not isinstance(payload, dict):
                continue
                
            part_name = payload.get('part_name', 'Unknown')
            oem_id = payload.get('oem_id', 'Unknown')
            quality = payload.get('quality_rating', 0)
            
            # Get supplier info
            suppliers = payload.get('suppliers', [])
            supplier_name = 'N/A'
            if suppliers and isinstance(suppliers, list) and len(suppliers) > 0:
                supplier = suppliers[0]
                if isinstance(supplier, dict):
                    supplier_details = supplier.get('supplier_details', {})
                    supplier_name = supplier_details.get('company_name', 'N/A')
            
            price = payload.get('price_usd', 0) or payload.get('unit_cost_usd', 0)
            lead_time = payload.get('lead_time_days', 0)
            
            details_table.add_row(
                part_name[:40] + "..." if len(part_name) > 40 else part_name,
                oem_id,
                f"{quality:.2f}",
                supplier_name[:20] + "..." if len(supplier_name) > 20 else supplier_name,
                f"${price:,.0f}" if price > 0 else "N/A",
                f"{lead_time:.0f} days" if lead_time > 0 else "N/A"
            )
        
        console.print(details_table)
        
        # Insight panel
        console.print("\n")
        console.print(Panel(
            "[bold]Cross-tenant search enables competitive benchmarking[/bold]\n\n"
            "• Single collection = simplified operations\n"
            "• No need for separate collections per tenant\n"
            "• Easy to compare offerings across OEMs\n"
            "• Unified query interface for all tenants",
            title="[green]Unified Architecture Benefits[/green]",
            border_style="green"
        ))
        
        demo_results["demos"]["demo2_cross_tenant"] = {
            "query": query2,
            "total_results": len(cross_tenant_results),
            "results_by_oem": {k: len(v) for k, v in results_by_oem.items()},
            "oem_analysis": oem_analysis,
            "top_results": [
                {
                    "part_name": r.payload.get('part_name') if hasattr(r, 'payload') and isinstance(r.payload, dict) else None,
                    "oem_id": r.payload.get('oem_id') if hasattr(r, 'payload') and isinstance(r.payload, dict) else None,
                    "quality_rating": r.payload.get('quality_rating') if hasattr(r, 'payload') and isinstance(r.payload, dict) else None,
                    "score": r.score if hasattr(r, 'score') else 0
                }
                for r in cross_tenant_results[:10]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in cross-tenant search: {e}")
        console.print(f"[red]✗ Error in cross-tenant search: {e}[/red]")
        demo_results["demos"]["demo2_cross_tenant"] = {"error": str(e)}
    
    # ========================================================================
    # DEMO 3: Tenant Promotion Scenario (Conceptual)
    # ========================================================================
    console.print("\n")
    console.print(Panel(
        "[bold yellow]DEMO 3: Tenant Promotion Scenario[/bold yellow]\n"
        "[dim]Simulating high-volume tenant promotion to dedicated shard[/dim]",
        border_style="yellow",
        title="Scaling with Tenant Promotion"
    ))
    console.print("\n")
    
    # Find highest volume OEM
    high_volume_oem = max(oems_data, key=lambda x: x.get('annual_production_units', 0))
    oem_id_high_volume = high_volume_oem.get('oem_id', 'OEM-C')
    production_units = high_volume_oem.get('annual_production_units', 0)
    
    console.print(f"[cyan]High-Volume Tenant:[/cyan] {high_volume_oem.get('company_name')} ({oem_id_high_volume})")
    console.print(f"[cyan]Annual Production:[/cyan] {production_units:,} units/year\n")
    
    console.print("[dim]Simulating query metrics for high-volume tenant...[/dim]\n")
    
    try:
        # Simulate query metrics (conceptual - actual metrics would come from monitoring)
        # In production, you'd query Qdrant metrics API or monitoring system
        query_metrics = {
            "oem_id": oem_id_high_volume,
            "avg_latency_shared_ms": 95.5,
            "peak_latency_shared_ms": 450.2,
            "avg_latency_dedicated_ms": 45.3,
            "peak_latency_dedicated_ms": 78.1,
            "queries_per_second": 1250,
            "noisy_neighbor_impact": "High - affects other tenants during peak"
        }
        
        # Display comparison table
        comparison_table = Table(title="Shared Shard vs Dedicated Shard", box=box.ROUNDED)
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Shared Shard", justify="right", style="yellow")
        comparison_table.add_column("Dedicated Shard", justify="right", style="green")
        
        comparison_table.add_row(
            "Avg Latency",
            f"{query_metrics['avg_latency_shared_ms']:.1f} ms",
            f"{query_metrics['avg_latency_dedicated_ms']:.1f} ms"
        )
        comparison_table.add_row(
            "Peak Latency",
            f"{query_metrics['peak_latency_shared_ms']:.1f} ms",
            f"{query_metrics['peak_latency_dedicated_ms']:.1f} ms"
        )
        comparison_table.add_row(
            "Isolation",
            "[yellow]Logical[/yellow]",
            "[green]Physical[/green]"
        )
        comparison_table.add_row(
            "Cost",
            "[green]Shared[/green]",
            "[yellow]Premium[/yellow]"
        )
        comparison_table.add_row(
            "Noisy Neighbor",
            "[red]Yes[/red]",
            "[green]No[/green]"
        )
        
        console.print(comparison_table)
        
        # Explanation panel
        console.print("\n")
        console.print(Panel(
            "[bold]The Noisy Neighbor Problem:[/bold]\n\n"
            "When a high-volume tenant shares resources with others:\n"
            "• Peak loads from one tenant affect all tenants\n"
            "• Unpredictable latency spikes\n"
            "• Difficult to guarantee SLAs\n\n"
            "[bold]Solution: Tenant Promotion[/bold]\n\n"
            "Qdrant 1.16 allows seamless promotion:\n"
            "• High-volume tenant moved to dedicated shard\n"
            "• Predictable performance guaranteed\n"
            "• No application code changes needed\n"
            "• Best of both worlds: start shared, promote when needed",
            title="[yellow]Tenant Promotion Explained[/yellow]",
            border_style="yellow"
        ))
        
        demo_results["demos"]["demo3_tenant_promotion"] = {
            "high_volume_oem": {
                "oem_id": oem_id_high_volume,
                "company_name": high_volume_oem.get('company_name'),
                "annual_production_units": production_units
            },
            "query_metrics": query_metrics,
            "promotion_benefits": {
                "latency_improvement": round(
                    (query_metrics['avg_latency_shared_ms'] - query_metrics['avg_latency_dedicated_ms']) / 
                    query_metrics['avg_latency_shared_ms'] * 100, 1
                ),
                "peak_latency_improvement": round(
                    (query_metrics['peak_latency_shared_ms'] - query_metrics['peak_latency_dedicated_ms']) / 
                    query_metrics['peak_latency_shared_ms'] * 100, 1
                )
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in tenant promotion demo: {e}")
        console.print(f"[red]✗ Error in tenant promotion demo: {e}[/red]")
        demo_results["demos"]["demo3_tenant_promotion"] = {"error": str(e)}
    
    # ========================================================================
    # DEMO 4: Hybrid Workload Example
    # ========================================================================
    console.print("\n")
    console.print(Panel(
        "[bold magenta]DEMO 4: Hybrid Workload Example[/bold magenta]\n"
        "[dim]OEM-A needs brake pads + competitive intelligence[/dim]",
        border_style="magenta",
        title="Practical Use Case"
    ))
    console.print("\n")
    
    scenario_oem = "OEM-A"
    query4 = "brake pads"
    
    console.print(f"[cyan]Scenario:[/cyan] {oems_dict.get(scenario_oem, {}).get('company_name', scenario_oem)} needs brake pads")
    console.print(f"[cyan]Requirement:[/cyan] Find internal options + competitive intelligence\n")
    
    try:
        # Step 1: Tenant-specific search
        console.print("[bold blue]Step 1: Tenant-Specific Search[/bold blue]")
        console.print(f"[dim]Searching {scenario_oem} parts only...[/dim]\n")
        
        tenant_specific_results = agent.search_parts_semantic(
            query=query4,
            filters={},
            oem_id=scenario_oem,
            limit=10
        )
        
        # Step 2: Cross-tenant search
        console.print("[bold green]Step 2: Cross-Tenant Search[/bold green]")
        console.print("[dim]Searching all OEMs for competitive intelligence...[/dim]\n")
        
        cross_tenant_results_4 = agent.search_parts_semantic(
            query=query4,
            filters={},
            oem_id=None,  # Cross-tenant
            limit=15
        )
        
        # Group cross-tenant results by OEM
        cross_by_oem = defaultdict(list)
        for result in cross_tenant_results_4:
            payload = result.payload if hasattr(result, 'payload') else {}
            if isinstance(payload, dict):
                oem_id = payload.get('oem_id', 'Unknown')
                if oem_id != scenario_oem:  # Exclude own OEM
                    cross_by_oem[oem_id].append(result)
        
        # Display side-by-side comparison
        comparison_table_4 = Table(title="Internal Options vs Market Options", box=box.ROUNDED)
        comparison_table_4.add_column("Source", style="cyan")
        comparison_table_4.add_column("Part Name", style="magenta")
        comparison_table_4.add_column("OEM", style="yellow")
        comparison_table_4.add_column("Quality", justify="right", style="green")
        comparison_table_4.add_column("Supplier", style="dim")
        
        # Add internal options
        comparison_table_4.add_row(
            "[blue]Internal[/blue]",
            "",
            "",
            "",
            ""
        )
        for result in tenant_specific_results[:5]:
            payload = result.payload if hasattr(result, 'payload') else {}
            if not isinstance(payload, dict):
                continue
                
            part_name = payload.get('part_name', 'Unknown')
            quality = payload.get('quality_rating', 0)
            
            suppliers = payload.get('suppliers', [])
            supplier_name = 'N/A'
            if suppliers and isinstance(suppliers, list) and len(suppliers) > 0:
                supplier = suppliers[0]
                if isinstance(supplier, dict):
                    supplier_details = supplier.get('supplier_details', {})
                    supplier_name = supplier_details.get('company_name', 'N/A')
            
            comparison_table_4.add_row(
                "",
                part_name[:35] + "..." if len(part_name) > 35 else part_name,
                scenario_oem,
                f"{quality:.2f}",
                supplier_name[:20] + "..." if len(supplier_name) > 20 else supplier_name
            )
        
        # Add market options
        comparison_table_4.add_row(
            "[green]Market[/green]",
            "",
            "",
            "",
            ""
        )
        for oem_id, oem_results in list(cross_by_oem.items())[:1]:  # Show one competitor
            for result in oem_results[:5]:
                payload = result.payload if hasattr(result, 'payload') else {}
                if not isinstance(payload, dict):
                    continue
                    
                part_name = payload.get('part_name', 'Unknown')
                quality = payload.get('quality_rating', 0)
                
                suppliers = payload.get('suppliers', [])
                supplier_name = 'N/A'
                if suppliers and isinstance(suppliers, list) and len(suppliers) > 0:
                    supplier = suppliers[0]
                    if isinstance(supplier, dict):
                        supplier_details = supplier.get('supplier_details', {})
                        supplier_name = supplier_details.get('company_name', 'N/A')
                
                comparison_table_4.add_row(
                    "",
                    part_name[:35] + "..." if len(part_name) > 35 else part_name,
                    oem_id,
                    f"{quality:.2f}",
                    supplier_name[:20] + "..." if len(supplier_name) > 20 else supplier_name
                )
        
        console.print(comparison_table_4)
        
        # Summary insights
        console.print("\n")
        console.print(Panel(
            "[bold]Hybrid Workload Advantages:[/bold]\n\n"
            "• [blue]Tenant-isolated search:[/blue] Find your own parts quickly\n"
            "• [green]Cross-tenant search:[/green] Discover competitor suppliers\n"
            "• [cyan]Unified architecture:[/cyan] Single query interface for both\n"
            "• [yellow]No code changes:[/yellow] Just toggle oem_id filter\n\n"
            "[dim]This demonstrates the flexibility of Qdrant 1.16 multitenancy - "
            "same collection, different access patterns.[/dim]",
            title="[magenta]Unified Architecture Flexibility[/magenta]",
            border_style="magenta"
        ))
        
        demo_results["demos"]["demo4_hybrid_workload"] = {
            "scenario_oem": scenario_oem,
            "query": query4,
            "tenant_specific_results": len(tenant_specific_results),
            "cross_tenant_results": len(cross_tenant_results_4),
            "competitor_oems_found": list(cross_by_oem.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid workload demo: {e}")
        console.print(f"[red]✗ Error in hybrid workload demo: {e}[/red]")
        demo_results["demos"]["demo4_hybrid_workload"] = {"error": str(e)}
    
    # ========================================================================
    # Summary and Best Practices
    # ========================================================================
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Summary: Qdrant 1.16 Multitenancy Benefits[/bold cyan]\n\n"
        "[bold]1. Tenant Isolation:[/bold]\n"
        "   • Complete data isolation using oem_id filters\n"
        "   • No data leakage between tenants\n"
        "   • Logical separation in single collection\n\n"
        "[bold]2. Unified Architecture:[/bold]\n"
        "   • Single collection easier than multiple collections\n"
        "   • Simplified operations and maintenance\n"
        "   • Cross-tenant queries enable benchmarking\n\n"
        "[bold]3. Scalability:[/bold]\n"
        "   • Start with shared shards (cost-effective)\n"
        "   • Promote high-volume tenants to dedicated shards\n"
        "   • No application code changes required\n\n"
        "[bold]4. Flexibility:[/bold]\n"
        "   • Tenant-specific queries (with oem_id filter)\n"
        "   • Cross-tenant queries (without oem_id filter)\n"
        "   • Hybrid workloads supported seamlessly",
        title="[cyan]Key Takeaways[/cyan]",
        border_style="cyan"
    ))
    
    console.print("\n")
    console.print(Panel(
        "[bold yellow]Best Practices[/bold yellow]\n\n"
        "• [green]Use oem_id filter[/green] for tenant-isolated queries\n"
        "• [green]Omit oem_id filter[/green] for cross-tenant analysis\n"
        "• [green]Monitor query patterns[/green] to identify promotion candidates\n"
        "• [green]Index oem_id field[/green] for efficient filtering\n"
        "• [green]Start shared[/green], promote when needed",
        title="[yellow]Tips[/yellow]",
        border_style="yellow"
    ))
    
    # Save results
    results_file = results_dir / "multitenancy_demo.json"
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to: {results_file}[/green]\n")
    
    return demo_results


if __name__ == "__main__":
    """
    Execute the multitenancy demonstration.
    
    This script demonstrates:
    - How Qdrant routes queries to correct tenants using oem_id filters
    - Why single collection is better than multiple collections
    - The "noisy neighbor" problem and tenant promotion solution
    - Practical hybrid workload examples
    """
    try:
        results = run_multitenancy_demo()
        console.print("\n[bold green]✓ Multitenancy demonstration completed successfully![/bold green]\n")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        console.print(f"\n[bold red]✗ Demo failed: {e}[/bold red]\n")
        raise

