# Automotive Supply Chain AI System

An AI-powered automotive supply chain management system using Qdrant 1.16 vector database and CrewAI multi-agent framework. Demonstrates advanced features including ACORN algorithm, tiered multitenancy, and enhanced full-text search.

## 🚀 Features

- **Semantic Parts Discovery**: Find automotive parts using natural language queries with Qdrant vector search
- **ACORN Algorithm**: Improved recall on restrictive filters for safety-critical parts
- **Tiered Multitenancy**: Multi-tenant data isolation with seamless tenant promotion
- **Enhanced Full-Text Search**: ASCII folding, MatchTextAny, and flexible text matching
- **Multi-Agent Workflows**: Coordinated AI agents for complex supply chain problem solving
- **Supplier Risk Assessment**: Comprehensive risk scoring and alternative supplier discovery

## 📋 Prerequisites

- Python 3.11 or higher
- Qdrant Cloud account (or self-hosted Qdrant 1.16+)
- OpenAI API key (for CrewAI agents)

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone git@github.com:iampique/ai_for_supply_chain.git
cd ai_for_supply_chain
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials. See `.env.example` for all available configuration options:

```env
# Required: Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-id.us-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key-here

# Required: OpenAI API Key (for CrewAI agents)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Collection Configuration (default: automotive_parts)
COLLECTION_NAME=automotive_parts

# Optional: Qdrant 1.16 Feature Settings
ACORN_ENABLED=true
ACORN_MAX_SELECTIVITY=0.4
ENABLE_MULTITENANCY=true
SHARD_NUMBER=3
```

**Get your credentials:**
- **Qdrant Cloud**: Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) and create a cluster
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

### 5. Load Data into Qdrant

```bash
python -m src.data_ingestion
```

This will:
- Load automotive parts data from JSON files
- Generate embeddings using sentence-transformers
- Upload to Qdrant Cloud with multitenancy enabled
- Create indexes for efficient search

## 🎯 Usage

### Run Demonstrations

#### ACORN Algorithm Comparison
```bash
python -m src.demo_acorn_comparison
```
Demonstrates Qdrant 1.16 ACORN algorithm performance across different filter selectivity scenarios.

#### Multitenancy Demonstration
```bash
python -m src.demo_multitenancy
```
Shows tenant isolation, cross-tenant search, and tenant promotion scenarios.

#### Full-Text Search Demo
```bash
python -m src.demo_fulltext_search
```
Demonstrates ASCII folding, MatchTextAny, and flexible text matching.

#### Multi-Agent Workflow
```bash
python -m src.demo_agentic_workflow
```
End-to-end demonstration of coordinated multi-agent workflow for crisis management.

### Use Agents Programmatically

```python
from src.agents.parts_discovery_agent import PartsDiscoveryAgent
from src.agents.supplier_alternative_agent import SupplierAlternativeAgent

# Initialize agents
parts_agent = PartsDiscoveryAgent()
supplier_agent = SupplierAlternativeAgent()

# Search for parts
results = parts_agent.search_parts_semantic(
    query="48V battery module with thermal management",
    oem_id="OEM-A",
    limit=10
)

# Find alternative suppliers
alternatives = supplier_agent.find_alternative_suppliers(
    part_id="DEN-0000001",
    min_quality_rating=4.5,
    max_results=5
)
```

## 📁 Project Structure

```
automotive-supply-chain-ai/
├── data/                          # Data files
│   ├── ev_hybrid_automotive_parts.json
│   ├── suppliers.json
│   ├── part_supplier_relationships.json
│   ├── oem_profiles.json
│   └── quality_incidents.json
├── src/                           # Source code
│   ├── agents/                    # AI agents
│   │   ├── parts_discovery_agent.py
│   │   └── supplier_alternative_agent.py
│   ├── config.py                  # Configuration management
│   ├── qdrant_client.py           # Qdrant client utilities
│   ├── data_loader.py             # Data loading and enrichment
│   ├── embeddings.py              # Embedding generation
│   ├── data_ingestion.py          # Data ingestion pipeline
│   └── demo_*.py                  # Demonstration scripts
├── results/                       # Generated outputs (gitignored)
│   ├── metrics/                   # Performance metrics
│   └── charts/                    # Visualizations
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                      # This file
```

## 🔑 Key Components

### PartsDiscoveryAgent
- Semantic search for automotive parts
- ACORN algorithm for restrictive filters
- Multitenancy support (OEM-specific searches)
- Full-text search capabilities

### SupplierAlternativeAgent
- Find alternative suppliers using vector search
- Comprehensive risk assessment
- Geographic diversification
- Quality and delivery metrics

### Qdrant 1.16 Features Demonstrated
- **ACORN**: Improved recall on low-selectivity filters
- **Tiered Multitenancy**: Tenant isolation with promotion capability
- **Enhanced Full-Text Search**: ASCII folding, MatchTextAny, stemming

## 📊 Data

The project includes sample automotive supply chain data:
- 500 automotive parts (EV/Hybrid components)
- 50 suppliers with detailed profiles
- 200 part-supplier relationships
- 3 OEM profiles (OEM-A, OEM-B, OEM-C)
- 250 quality incidents

## 🔒 Security

**Important**: Never commit your `.env` file to version control. It contains sensitive API keys.

The `.gitignore` file is configured to exclude:
- `.env` files
- Virtual environment (`venv/`)
- Generated results (`results/`)
- Python cache files (`__pycache__/`)

## 🧪 Testing

Verify your environment setup:

```bash
# Test imports
python -c "from src.agents.parts_discovery_agent import PartsDiscoveryAgent; print('✓ OK')"

# Test Qdrant connection
python -c "from src.qdrant_client import get_qdrant_client; get_qdrant_client(); print('✓ Connected')"

# Test all components
python -c "from src.agents.parts_discovery_agent import PartsDiscoveryAgent; from src.agents.supplier_alternative_agent import SupplierAlternativeAgent; from src.qdrant_client import get_qdrant_client; get_qdrant_client(); print('✓ All components ready')"
```

For detailed testing instructions, see **TEST_WORKFLOW.md**.

## 📝 Documentation

- **TEST_WORKFLOW.md**: Detailed testing guide for agentic workflows

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Qdrant for the powerful vector database
- CrewAI for the multi-agent framework
- Sentence Transformers for embeddings

## 📧 Support

For issues and questions, please open an issue on GitHub.
