# LLM, AI Agent & RAG Learning Project

A comprehensive learning project to master Language Models, AI Agents, and Retrieval Augmented Generation (RAG).

## 📁 Project Structure

```
llm_ai_agent_rag/
├── 01_llm_basics/                 # Core LLM concepts
│   ├── simple_llm.py              # Basic LLM integration
│   ├── prompts/
│   │   ├── system_prompts.py      # Predefined system prompts
│   │   └── few_shot_examples.py   # Few-shot learning examples
│   ├── models/
│   │   ├── model_config.py        # Model configurations
│   │   └── embedding_model.py     # Embedding models for RAG
│   └── api_integration/
│       ├── openai_client.py       # OpenAI API wrapper
│       └── anthropic_client.py    # Claude API wrapper
│
├── 02_ai_agents/                  # Agent systems
│   ├── simple_agent.py            # Basic agent implementation
│   ├── simple_agent/              # Advanced agent examples
│   ├── advanced_agent/            # Complex agent patterns
│   └── tools/                     # Reusable tools for agents
│
├── 03_rag_system/                 # RAG implementation
│   ├── rag_pipeline.py            # Main RAG pipeline
│   ├── embeddings/                # Embedding models
│   ├── vector_store/              # Vector database wrappers
│   ├── retrieval/                 # Retrieval strategies
│   └── indexing/                  # Document indexing
│
├── 04_interview_prep/             # Interview materials
│   ├── interview_questions.md     # Q&A guide
│   ├── solutions/                 # Code solutions
│   └── projects/                  # Practice projects
│
├── data/                          # Sample datasets
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── .env                           # Your API keys (don't commit!)
├── .env.example                   # Template for .env
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Step 1: Setup Environment

```bash
cd /Users/faye.pasvouri/data_engineering_zoomcamp_prep/llm_ai_agent_rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required keys**:
- `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com/ (optional)
- `CHROMA_DB_PATH`: Local path for vector store (already set)

### Step 4: Verify Setup

```bash
# Test OpenAI client
python 01_llm_basics/simple_llm.py

# Test agent
python 02_ai_agents/simple_agent.py

# Test RAG
python 03_rag_system/rag_pipeline.py
```

## 📚 Learning Path

### Week 1: LLM Basics
1. Study `01_llm_basics/simple_llm.py`
2. Learn about prompts in `prompts/system_prompts.py`
3. Understand model configs in `models/model_config.py`
4. Practice with API integrations

### Week 2: Embeddings & Retrieval
1. Study `models/embedding_model.py`
2. Learn semantic search
3. Understand vector databases
4. Practice with sample documents

### Week 3: RAG Systems
1. Study `03_rag_system/rag_pipeline.py`
2. Implement document indexing
3. Build retrieval strategies
4. Combine with LLM generation

### Week 4: AI Agents
1. Study `02_ai_agents/simple_agent.py`
2. Learn about tools and memory
3. Build multi-step agents
4. Practice decision-making

### Week 5: Interview Prep
1. Review `04_interview_prep/interview_questions.md`
2. Practice coding problems
3. Study system design
4. Mock interviews

## 🎯 Key Concepts

### Language Models (LLM)
- Predict next token in sequence
- Transformer architecture with attention
- Fine-tuning and prompt engineering
- Temperature, top_p, max_tokens parameters

### Embeddings
- Dense vector representations of text
- Semantic similarity matching
- Pre-trained models (e.g., SBERT)
- Used in RAG for retrieval

### RAG (Retrieval Augmented Generation)
1. **Retrieve**: Find relevant documents
2. **Augment**: Add context to prompt
3. **Generate**: LLM produces answer
4. **Result**: More accurate, factual responses

### Agents
- Autonomous systems that use tools
- Decision-making based on observations
- Memory for context
- Tool registration and execution

## 💻 Usage Examples

### Simple LLM Query
```python
from llm_ai_agent_rag.api_integration.openai_client import OpenAIClient

client = OpenAIClient()
response = client.simple_prompt("What is machine learning?")
print(response)
```

### Embedding & Search
```python
from llm_ai_agent_rag.models.embedding_model import EmbeddingModel

model = EmbeddingModel()
documents = ["Data engineering is about pipelines", "ML needs data"]
results = model.semantic_search("data processing", documents)
```

### RAG Pipeline
```python
from llm_ai_agent_rag.rag_system.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_documents(["Document 1", "Document 2"])
result = rag.query("What is this about?")
```

### Simple Agent
```python
from llm_ai_agent_rag.simple_agent import SimpleAgent

agent = SimpleAgent("MyAgent")
agent.register_tool("calculator", lambda q: "calculated")
response = agent.execute("Calculate 5+3")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_llm.py

# Run with coverage
pytest --cov=llm_ai_agent_rag tests/
```

## 📊 Project Status

- [ ] LLM basics implementation
- [ ] Embedding models setup
- [ ] RAG pipeline complete
- [ ] Agent system working
- [ ] Interview prep materials
- [ ] Comprehensive tests
- [ ] Documentation

## 🔗 Resources

### Learning Materials
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Vector Databases Guide](https://www.pinecone.io/learn/)

### Tools & Libraries
- **OpenAI**: `openai`
- **Anthropic**: `anthropic`
- **Embeddings**: `sentence-transformers`
- **Vector DB**: `chromadb`, `pinecone`
- **Data Processing**: `pandas`, `numpy`

### Interview Prep
- See `04_interview_prep/interview_questions.md`
- Practice LeetCode problems
- Study system design patterns
- Read engineering blogs

## ⚠️ Important Notes

1. **API Keys Security**: Never commit `.env` file to Git
2. **Rate Limits**: Monitor API usage to avoid unexpected charges
3. **Data Privacy**: Don't send sensitive data to APIs
4. **Free Tier**: Start with free tiers to learn
5. **Cost**: OpenAI GPT-4 is expensive, use GPT-3.5 for learning

## 🤝 Contributing

Improvements welcome! Please:
1. Create a new branch
2. Make your changes
3. Add tests
4. Create a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### ImportError: No module named 'openai'
```bash
pip install openai
```

### OPENAI_API_KEY not found
```bash
# Make sure .env file exists and has the key
ls -la .env
echo "OPENAI_API_KEY=sk-xxx" >> .env
```

### Connection errors
- Check internet connection
- Verify API key is valid
- Check API rate limits
- Look at error logs

## 👋 Getting Help

- Check `04_interview_prep/interview_questions.md` for concepts
- Review example code in each module
- Read official documentation links
- Test with simple examples first

---

**Happy Learning! 🎓**

For questions or suggestions, feel free to explore and experiment with the code!
