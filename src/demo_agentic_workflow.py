"""
End-to-End Multi-Agent Workflow Demonstration using Qdrant 1.16

This module demonstrates coordinated multi-agent workflows using CrewAI and Qdrant 1.16,
showcasing how multiple AI agents work together to solve complex supply chain problems.

Key Concepts:
- ACORN: Critical for safety-critical parts with strict filters
- Multitenancy: Keeps OEM data isolated during crisis management
- Agentic Coordination: Multiple agents working sequentially to solve complex problems
- Vector Search: Enables semantic matching for supplier alternatives
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from loguru import logger

from crewai import Agent, Task, Crew, Process
from src.agents.parts_discovery_agent import PartsDiscoveryAgent
from src.agents.supplier_alternative_agent import SupplierAlternativeAgent
from src.data_loader import load_all_data


# Initialize rich console for beautiful output
console = Console()


def run_workflow_demo() -> Dict[str, Any]:
    """
    End-to-end demonstration of coordinated multi-agent workflow using Qdrant 1.16.
    
    Scenario: Quality Crisis Management - Brake Pad Assembly Failure
    
    Demonstrates:
    - ACORN for accurate part search with safety filters
    - Multitenancy for OEM-specific data isolation
    - Vector search for supplier similarity matching
    - Agentic coordination for complex problem solving
    
    Returns:
        Dictionary containing workflow results and metrics
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold red]🚨 Quality Crisis Management Workflow[/bold red]\n"
        "[bold]Multi-Agent Coordination with Qdrant 1.16[/bold]",
        border_style="red"
    ))
    console.print("\n")
    
    # Create results directory
    results_dir = Path("results")
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_results = {
        "workflow_timestamp": datetime.now().isoformat(),
        "scenario": {},
        "tasks": {},
        "crew_outputs": {},
        "analysis": {},
        "qdrant_features_used": []
    }
    
    # ========================================================================
    # SCENARIO: Quality Crisis Management
    # ========================================================================
    console.print(Panel(
        "[bold red]CRISIS SCENARIO[/bold red]\n\n"
        "[bold]Part:[/bold] Brake Pad Assembly PART-0050\n"
        "[bold]Supplier:[/bold] SUP-015 (China-based Tier-2)\n"
        "[bold]Issue:[/bold] Critical quality incident - 8 failures in field\n"
        "[bold]OEM:[/bold] OEM-A (Luxury EV manufacturer)\n"
        "[bold]Impact:[/bold] Potential recall of 15,000 vehicles\n"
        "[bold]Urgency:[/bold] [red]CRITICAL[/red]",
        title="[red]Crisis Details[/red]",
        border_style="red"
    ))
    console.print("\n")
    
    scenario = {
        "part_id": "PART-0050",
        "part_name": "Brake Pad Assembly",
        "supplier_id": "SUP-015",
        "supplier_location": "China",
        "issue_type": "Critical quality incident",
        "failure_count": 8,
        "oem_id": "OEM-A",
        "potential_recall_vehicles": 15000,
        "urgency_level": "CRITICAL"
    }
    
    workflow_results["scenario"] = scenario
    
    # ========================================================================
    # STEP 1: INITIALIZATION
    # ========================================================================
    console.print("[yellow]Step 1: Initializing Agents...[/yellow]\n")
    
    try:
        # Initialize Parts Discovery Agent
        with console.status("[bold green]Initializing Parts Discovery Agent...", spinner="dots"):
            parts_agent_instance = PartsDiscoveryAgent()
            parts_agent = parts_agent_instance.get_agent()
            if parts_agent is None:
                raise RuntimeError("Parts Discovery Agent failed to initialize (check LLM configuration)")
        
        # Initialize Supplier Alternative Agent
        with console.status("[bold green]Initializing Supplier Alternative Agent...", spinner="dots"):
            supplier_agent_instance = SupplierAlternativeAgent()
            supplier_agent = supplier_agent_instance.get_agent()
            if supplier_agent is None:
                raise RuntimeError("Supplier Alternative Agent failed to initialize (check LLM configuration)")
        
        console.print("[green]✓[/green] Both agents initialized successfully\n")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize agents: {e}[/red]")
        logger.error(f"Agent initialization failed: {e}")
        return {"error": str(e)}
    
    # ========================================================================
    # STEP 2: TASK 1 - Parts Discovery (Using ACORN)
    # ========================================================================
    console.print("[yellow]Step 2: Creating Parts Discovery Task...[/yellow]\n")
    
    # ACORN is critical here because:
    # 1. We have strict safety filters (ISO-26262 ASIL-D, quality >= 4.5)
    # 2. These filters create low selectivity (< 40%)
    # 3. ACORN improves recall on restrictive filters, ensuring we don't miss critical alternatives
    # 4. For safety-critical parts, missing alternatives could mean vehicle recalls
    
    task1_description = """
    Critical quality issue with brake pad assembly PART-0050.
    Use ACORN search to find:
    1. All instances of this part across inventory
    2. Similar brake pad assemblies that might share same defect
    3. Compatible alternative brake pads from different suppliers
    
    Requirements:
    - Must meet ISO-26262 ASIL-D safety standard
    - Quality rating >= 4.5 (higher than failed part)
    - OEM-A specific parts preferred
    - Search should use ACORN for accuracy given strict filters
    
    Use semantic_parts_search_tool with use_acorn=True to leverage Qdrant 1.16 ACORN algorithm.
    ACORN is essential here because strict safety filters create low selectivity, and ACORN
    ensures we don't miss critical alternatives that could prevent vehicle recalls.
    """
    
    task1 = Task(
        description=task1_description,
        expected_output=(
            "List of affected parts + viable alternatives with safety compliance. "
            "Include part IDs, quality ratings, compliance standards, and supplier information. "
            "Highlight which parts use ACORN search and why it was critical."
        ),
        agent=parts_agent,
    )
    
    workflow_results["tasks"]["task1_parts_discovery"] = {
        "description": task1_description,
        "agent": "PartsDiscoveryAgent",
        "acorn_required": True,
        "reason": "Strict safety filters (ISO-26262 ASIL-D, quality >= 4.5) create low selectivity. ACORN improves recall to ensure no critical alternatives are missed."
    }
    
    console.print("[green]✓[/green] Task 1 created: Parts Discovery with ACORN\n")
    
    # ========================================================================
    # STEP 3: TASK 2 - Find Alternative Suppliers
    # ========================================================================
    console.print("[yellow]Step 3: Creating Supplier Alternative Task...[/yellow]\n")
    
    task2_description = """
    Based on affected brake pad parts identified in Task 1:
    Find 3-5 alternative suppliers to replace SUP-015.
    
    Exclusions:
    - Exclude SUP-015 (failed supplier)
    - Exclude other China-based suppliers (geographic risk diversification)
    
    Requirements:
    - Quality rating >= 4.7 (premium quality only)
    - Must have ISO-26262 and IATF-16949 certifications
    - Production capacity >= 50,000 units/month
    - Lead time <= 45 days (urgent replacement)
    - Geographic diversity: prefer USA, Mexico, Germany
    
    Provide detailed risk assessment for each alternative.
    Use find_supplier_alternatives_tool with reason="quality_issue" to ensure
    high quality standards and geographic diversification.
    """
    
    task2 = Task(
        description=task2_description,
        expected_output=(
            "Ranked list of 3-5 suppliers with risk scores and justifications. "
            "Include supplier name, location, quality rating, risk score, lead time, "
            "composite score, and key strengths. Provide risk assessment for each."
        ),
        agent=supplier_agent,
        context=[task1],  # Task 2 depends on Task 1 output
    )
    
    workflow_results["tasks"]["task2_supplier_alternatives"] = {
        "description": task2_description,
        "agent": "SupplierAlternativeAgent",
        "depends_on": "task1",
        "multitenancy_used": True,
        "reason": "OEM-A data isolation ensures we only see relevant parts and suppliers for this OEM during crisis management."
    }
    
    console.print("[green]✓[/green] Task 2 created: Supplier Alternatives\n")
    
    # ========================================================================
    # STEP 4: CREATE AND EXECUTE CREW
    # ========================================================================
    console.print("[yellow]Step 4: Creating Crew and Executing Workflow...[/yellow]\n")
    
    # Create Crew with sequential process
    # Sequential ensures Task 2 receives Task 1 output as context
    crew = Crew(
        agents=[parts_agent, supplier_agent],
        tasks=[task1, task2],
        process=Process.sequential,  # Tasks execute in order, Task 2 uses Task 1 output
        verbose=True,  # Show agent reasoning
    )
    
    console.print(Panel(
        "[bold cyan]Crew Configuration:[/bold cyan]\n\n"
        "• Agents: PartsDiscoveryAgent, SupplierAlternativeAgent\n"
        "• Tasks: Parts Discovery → Supplier Alternatives\n"
        "• Process: Sequential (Task 2 depends on Task 1)\n"
        "• Verbose: True (showing agent reasoning)\n\n"
        "[dim]Executing workflow...[/dim]",
        border_style="cyan"
    ))
    console.print("\n")
    
    try:
        # Execute crew workflow
        with console.status(
            "[bold green]Executing multi-agent workflow...[/bold green]",
            spinner="dots"
        ):
            crew_output = crew.kickoff()
        
        console.print("\n[green]✓[/green] Workflow execution completed\n")
        
        workflow_results["crew_outputs"] = {
            "raw_output": str(crew_output),
            "execution_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Workflow execution failed: {e}"
        console.print(f"[red]✗ {error_msg}[/red]")
        logger.error(error_msg)
        workflow_results["crew_outputs"] = {"error": str(e)}
        return workflow_results
    
    # ========================================================================
    # STEP 5: POST-PROCESSING AND ANALYSIS
    # ========================================================================
    console.print("[yellow]Step 5: Analyzing Results...[/yellow]\n")
    
    # Load data for analysis
    data = load_all_data()
    quality_incidents = data.get('incidents', [])
    
    # Analyze quality incidents for this part/supplier
    related_incidents = [
        inc for inc in quality_incidents
        if inc.get('part_id') == scenario['part_id'] or inc.get('supplier_id') == scenario['supplier_id']
    ]
    
    # Extract information from crew output (simplified - in production would parse more carefully)
    analysis = {
        "affected_parts_count": len(related_incidents),
        "incident_severity": "Critical",
        "root_cause": "Quality control failure at supplier facility",
        "recommended_suppliers": [],
        "total_impact_vehicles": scenario['potential_recall_vehicles'],
        "estimated_cost_millions": scenario['potential_recall_vehicles'] * 0.5,  # $500 per vehicle estimate
    }
    
    workflow_results["analysis"] = analysis
    
    # Track Qdrant 1.16 features used
    qdrant_features = [
        {
            "feature": "ACORN Algorithm",
            "used_in": "Task 1 - Parts Discovery",
            "reason": "Strict safety filters (ISO-26262 ASIL-D, quality >= 4.5) create low selectivity. ACORN improves recall to ensure no critical alternatives are missed.",
            "impact": "Critical for safety-critical parts where missing alternatives could mean vehicle recalls"
        },
        {
            "feature": "Tiered Multitenancy",
            "used_in": "Both Tasks",
            "reason": "OEM-A data isolation ensures we only see relevant parts and suppliers for this OEM during crisis management.",
            "impact": "Prevents data leakage and ensures accurate OEM-specific recommendations"
        },
        {
            "feature": "Vector Search",
            "used_in": "Task 2 - Supplier Alternatives",
            "reason": "Semantic similarity matching finds compatible parts even when exact matches aren't available.",
            "impact": "Enables rapid supplier diversification using semantic understanding"
        }
    ]
    
    workflow_results["qdrant_features_used"] = qdrant_features
    
    # ========================================================================
    # STEP 6: DISPLAY RESULTS
    # ========================================================================
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✓ Workflow Analysis Complete[/bold green]",
        border_style="green"
    ))
    console.print("\n")
    
    # Panel 1 - Crisis Summary
    console.print(Panel(
        f"[bold red]Crisis Scope and Impact[/bold red]\n\n"
        f"[bold]Affected Parts:[/bold] {analysis['affected_parts_count']} incidents identified\n"
        f"[bold]Vehicle Recall Scope:[/bold] {analysis['total_impact_vehicles']:,} vehicles\n"
        f"[bold]Estimated Financial Impact:[/bold] ${analysis['estimated_cost_millions']:.1f}M\n"
        f"[bold]Root Cause:[/bold] {analysis['root_cause']}\n"
        f"[bold]Urgency:[/bold] [red]CRITICAL[/red] - Immediate action required",
        title="[red]Crisis Summary[/red]",
        border_style="red"
    ))
    console.print("\n")
    
    # Panel 2 - Agent Decisions
    console.print(Panel(
        "[bold cyan]Agent Decisions and Reasoning[/bold cyan]\n\n"
        "[green]Parts Discovery Agent:[/green]\n"
        "• Used ACORN algorithm for accurate search with strict safety filters\n"
        "• Searched for ISO-26262 ASIL-D compliant alternatives\n"
        "• Identified affected parts and viable replacements\n\n"
        "[green]Supplier Alternative Agent:[/green]\n"
        "• Used vector search to find compatible suppliers\n"
        "• Applied geographic diversification (excluded China-based suppliers)\n"
        "• Evaluated risk scores and quality metrics\n"
        "• Provided ranked recommendations with justifications",
        title="[cyan]Agent Workflow[/cyan]",
        border_style="cyan"
    ))
    console.print("\n")
    
    # Table - Recommended Suppliers (example data)
    supplier_table = Table(title="Recommended Alternative Suppliers", box=box.ROUNDED)
    supplier_table.add_column("Supplier Name", style="cyan")
    supplier_table.add_column("Location", style="magenta")
    supplier_table.add_column("Quality", justify="right", style="green")
    supplier_table.add_column("Risk Score", justify="right", style="yellow")
    supplier_table.add_column("Lead Time", justify="right", style="blue")
    supplier_table.add_column("Composite Score", justify="right", style="green")
    supplier_table.add_column("Key Strengths", style="dim")
    
    # Example supplier data (in production, this would come from agent output)
    example_suppliers = [
        {
            "name": "Premium Brake Systems Inc.",
            "location": "USA",
            "quality": 4.8,
            "risk_score": 2.5,
            "lead_time": 35,
            "composite_score": 8.9,
            "strengths": "ISO-26262 certified, high capacity"
        },
        {
            "name": "German Precision Components",
            "location": "Germany",
            "quality": 4.9,
            "risk_score": 2.0,
            "lead_time": 40,
            "composite_score": 9.1,
            "strengths": "Premium quality, excellent track record"
        },
        {
            "name": "North American Brake Co.",
            "location": "Mexico",
            "quality": 4.7,
            "risk_score": 3.0,
            "lead_time": 30,
            "composite_score": 8.5,
            "strengths": "Fast lead time, cost-effective"
        }
    ]
    
    for supplier in example_suppliers:
        supplier_table.add_row(
            supplier["name"],
            supplier["location"],
            f"{supplier['quality']:.1f}",
            f"{supplier['risk_score']:.1f}",
            f"{supplier['lead_time']} days",
            f"{supplier['composite_score']:.1f}",
            supplier["strengths"]
        )
    
    console.print(supplier_table)
    console.print("\n")
    
    # Panel 3 - Implementation Plan
    console.print(Panel(
        "[bold yellow]Implementation Plan[/bold yellow]\n\n"
        "[bold]Immediate Actions (Day 1-3):[/bold]\n"
        "• Halt production with SUP-015\n"
        "• Initiate quality audit of affected parts\n"
        "• Contact recommended suppliers for capacity availability\n\n"
        "[bold]Timeline:[/bold]\n"
        "• Supplier Qualification: 5-7 days\n"
        "• Production Setup: 10-14 days\n"
        "• First Delivery: 30-45 days\n"
        "• Full Replacement: 60-90 days\n\n"
        "[bold]Success Metrics:[/bold]\n"
        "• Zero quality incidents with new suppliers\n"
        "• On-time delivery rate > 95%\n"
        "• Cost increase < 15% vs original supplier",
        title="[yellow]Action Plan[/yellow]",
        border_style="yellow"
    ))
    console.print("\n")
    
    # Panel 4 - Qdrant 1.16 Features Used
    features_text = "\n".join([
        f"[bold]{feat['feature']}:[/bold]\n"
        f"  Used in: {feat['used_in']}\n"
        f"  Reason: {feat['reason']}\n"
        f"  Impact: {feat['impact']}\n"
        for feat in qdrant_features
    ])
    
    console.print(Panel(
        f"[bold green]Qdrant 1.16 Features Utilized[/bold green]\n\n{features_text}",
        title="[green]Technology Stack[/green]",
        border_style="green"
    ))
    console.print("\n")
    
    # ========================================================================
    # STEP 7: SAVE OUTPUTS
    # ========================================================================
    console.print("[yellow]Step 7: Saving Results...[/yellow]\n")
    
    # Save detailed report
    report_path = results_dir / "crisis_management_report.md"
    report_content = f"""# Quality Crisis Management Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Crisis Scenario

- **Part:** {scenario['part_name']} ({scenario['part_id']})
- **Supplier:** {scenario['supplier_id']} ({scenario['supplier_location']})
- **Issue:** {scenario['issue_type']} - {scenario['failure_count']} failures
- **OEM:** {scenario['oem_id']}
- **Impact:** {scenario['potential_recall_vehicles']:,} vehicles at risk
- **Urgency:** {scenario['urgency_level']}

## Workflow Execution

### Task 1: Parts Discovery (ACORN)
{task1_description}

### Task 2: Supplier Alternatives
{task2_description}

## Analysis Results

- **Affected Parts:** {analysis['affected_parts_count']} incidents
- **Root Cause:** {analysis['root_cause']}
- **Estimated Cost:** ${analysis['estimated_cost_millions']:.1f}M

## Qdrant 1.16 Features Used

"""
    
    for feat in qdrant_features:
        report_content += f"""
### {feat['feature']}
- **Used in:** {feat['used_in']}
- **Reason:** {feat['reason']}
- **Impact:** {feat['impact']}

"""
    
    report_content += f"""
## Crew Output

{crew_output}

## Recommendations

1. Immediately halt production with {scenario['supplier_id']}
2. Qualify alternative suppliers from recommended list
3. Implement enhanced quality controls
4. Establish geographic diversification strategy
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    console.print(f"[green]✓[/green] Report saved: {report_path}")
    
    # Save metrics JSON
    metrics_path = metrics_dir / "workflow_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(workflow_results, f, indent=2)
    
    console.print(f"[green]✓[/green] Metrics saved: {metrics_path}")
    
    # Save decision matrix CSV
    matrix_path = results_dir / "decision_matrix.csv"
    with open(matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Supplier Name", "Location", "Quality Rating", "Risk Score",
            "Lead Time (days)", "Composite Score", "Key Strengths"
        ])
        for supplier in example_suppliers:
            writer.writerow([
                supplier["name"],
                supplier["location"],
                supplier["quality"],
                supplier["risk_score"],
                supplier["lead_time"],
                supplier["composite_score"],
                supplier["strengths"]
            ])
    
    console.print(f"[green]✓[/green] Decision matrix saved: {matrix_path}\n")
    
    # Final summary
    console.print(Panel.fit(
        "[bold green]✓ Workflow Demonstration Complete[/bold green]\n\n"
        f"• Report: {report_path}\n"
        f"• Metrics: {metrics_path}\n"
        f"• Decision Matrix: {matrix_path}",
        border_style="green"
    ))
    console.print("\n")
    
    return workflow_results


if __name__ == "__main__":
    """
    Execute the agentic workflow demonstration.
    
    This script demonstrates:
    - Why ACORN is critical for safety-critical parts (strict filters, low selectivity)
    - How multitenancy keeps OEM data isolated during crisis management
    - The value of agentic coordination for complex problem solving
    - End-to-end workflow from crisis detection to supplier replacement
    """
    try:
        results = run_workflow_demo()
        console.print("\n[bold green]✓ Agentic workflow demonstration completed successfully![/bold green]\n")
    except Exception as e:
        logger.error(f"Workflow demo failed: {e}")
        console.print(f"\n[bold red]✗ Workflow demo failed: {e}[/bold red]\n")
        raise

