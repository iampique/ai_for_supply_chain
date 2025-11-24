# Testing Guide: Agentic Workflow Demo

## Prerequisites

1. **Virtual Environment**: Make sure you're in the virtual environment
   ```bash
   # If not activated:
   source venv/bin/activate
   # You should see (venv) in your prompt
   ```

2. **OpenAI API Key**: Required for CrewAI agents to function
   ```bash
   # Check if set:
   echo $OPENAI_API_KEY
   
   # If not set, add to .env file:
   # OPENAI_API_KEY=your-api-key-here
   ```

3. **Qdrant Connection**: Verify Qdrant Cloud is accessible
   ```bash
   python -c "from src.qdrant_client import get_qdrant_client; client = get_qdrant_client(); print('✓ Qdrant OK')"
   ```

## Step-by-Step Testing

### Step 1: Verify Environment Setup

```bash
# Navigate to project directory (adjust path as needed)
cd ai_for_supply_chain

# Activate virtual environment (if not already active)
source venv/bin/activate

# Verify Python path
which python
# Should show: .../venv/bin/python
```

### Step 2: Check Dependencies

```bash
# Test CrewAI imports
python -c "from crewai import Crew, Task, Process; print('✓ CrewAI OK')"

# Test agent imports
python -c "from src.agents.parts_discovery_agent import PartsDiscoveryAgent; print('✓ PartsDiscoveryAgent OK')"
python -c "from src.agents.supplier_alternative_agent import SupplierAlternativeAgent; print('✓ SupplierAlternativeAgent OK')"

# Test Qdrant connection
python -c "from src.qdrant_client import get_qdrant_client; client = get_qdrant_client(); print('✓ Qdrant connection OK')"
```

### Step 3: Verify OpenAI API Key

```bash
# Check if API key is set
python -c "import os; key = os.getenv('OPENAI_API_KEY'); print('OPENAI_API_KEY:', 'SET ✓' if key else 'NOT SET ✗')"

# If not set, check .env file
cat .env | grep OPENAI_API_KEY
```

**Important**: If the API key is not set, the agents will fail to initialize. Add it to your `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### Step 4: Run the Workflow Demo

```bash
# Execute the demo
python -m src.demo_agentic_workflow
```

**Expected Execution Flow:**
1. **Crisis Scenario Display** (red panel with crisis details)
2. **Agent Initialization** (PartsDiscoveryAgent, SupplierAlternativeAgent)
3. **Task Creation** (Task 1: Parts Discovery, Task 2: Supplier Alternatives)
4. **Crew Execution** (this may take 30-60 seconds as agents make LLM calls)
5. **Results Display** (crisis summary, agent decisions, supplier recommendations)
6. **File Generation** (report, metrics, decision matrix)

### Step 5: Monitor Execution

During execution, you'll see:
- **Status indicators**: `[yellow]Step X: ...[/yellow]`
- **Progress spinners**: `[bold green]Executing...[/bold green]`
- **Agent reasoning**: CrewAI verbose output showing agent thoughts
- **Success indicators**: `[green]✓[/green]` for completed steps

**Note**: The crew execution step may take 30-60 seconds as it:
- Makes LLM API calls to OpenAI
- Processes agent reasoning
- Executes tool calls (Qdrant searches)

### Step 6: Verify Generated Files

After execution completes, check the outputs:

```bash
# Check report
ls -lh results/crisis_management_report.md
cat results/crisis_management_report.md | head -30

# Check metrics
ls -lh results/metrics/workflow_metrics.json
cat results/metrics/workflow_metrics.json | python -m json.tool | head -50

# Check decision matrix
ls -lh results/decision_matrix.csv
cat results/decision_matrix.csv
```

### Step 7: Review Output

The console output should show:

1. **Crisis Summary Panel** (red)
   - Affected parts count
   - Vehicle recall scope
   - Financial impact

2. **Agent Decisions Panel** (cyan)
   - Parts Discovery Agent findings
   - Supplier Alternative Agent recommendations

3. **Supplier Recommendations Table**
   - Supplier names, locations, quality ratings
   - Risk scores, lead times, composite scores

4. **Implementation Plan Panel** (yellow)
   - Immediate actions
   - Timeline
   - Success metrics

5. **Qdrant Features Panel** (green)
   - ACORN Algorithm usage
   - Multitenancy usage
   - Vector search usage

## Troubleshooting

### Issue: "Failed to initialize agents"

**Solution**: Check OpenAI API key
```bash
# Verify API key is set
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT SET'))"

# If not set, add to .env and reload
export $(cat .env | grep OPENAI_API_KEY | xargs)
```

### Issue: "Qdrant connection failed"

**Solution**: Verify Qdrant credentials in `.env`
```bash
cat .env | grep QDRANT
```

### Issue: "ModuleNotFoundError"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Crew execution hangs or takes too long

**Solution**: This is normal - LLM calls take time
- Wait 30-60 seconds
- Check your OpenAI API quota
- Verify internet connection

### Issue: No results generated

**Solution**: Check if execution completed
```bash
# Check for error messages in output
# Verify agents initialized successfully
# Check OpenAI API key is valid
```

## Quick Test Command

Run this single command to test everything:

```bash
cd ai_for_supply_chain && \
source venv/bin/activate && \
python -c "import os; assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not set'; from crewai import Crew; from src.agents.parts_discovery_agent import PartsDiscoveryAgent; from src.qdrant_client import get_qdrant_client; get_qdrant_client(); print('✓ All checks passed')" && \
python -m src.demo_agentic_workflow
```

## Expected Runtime

- **Agent Initialization**: 5-10 seconds
- **Task Creation**: < 1 second
- **Crew Execution**: 30-60 seconds (LLM calls)
- **Post-processing**: 1-2 seconds
- **Total**: ~1-2 minutes

## Success Indicators

✅ **Successful execution shows:**
- Both agents initialized
- Tasks created successfully
- Crew execution completed
- Results displayed in formatted panels
- Files generated in `results/` directory
- Final success message: "✓ Workflow Demonstration Complete"

## Next Steps

After successful execution:
1. Review `results/crisis_management_report.md` for detailed analysis
2. Check `results/metrics/workflow_metrics.json` for workflow metrics
3. Examine `results/decision_matrix.csv` for supplier comparison
4. Review console output for agent reasoning traces

