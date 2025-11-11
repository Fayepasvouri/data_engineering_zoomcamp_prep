# 🚀 Quick Start Guide

## 5 Minutes to Get Running

### Step 1: Install Dependencies (2 min)
```bash
cd /Users/faye.pasvouri/data_engineering_zoomcamp_prep/llm_ai_agent_rag
pip install -r requirements.txt
```

### Step 2: Verify API Key (1 min)
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Should see: OPENAI_API_KEY=sk-proj-...
```

### Step 3: Run First Example (2 min)
```bash
python 01_llm_basics/simple_llm.py
```

### Step 4: Explore More
```bash
# Run agent example
python 02_ai_agents/simple_agent.py

# Run RAG example
python 03_rag_system/rag_pipeline.py
```

---

## 📖 What to Learn First

1. **Read the Main README**
   ```bash
   cat README.md
   ```

2. **Study Interview Guide**
   ```bash
   cat 04_interview_prep/interview_questions.md
   ```

3. **Review Python Files**
   - Start: `01_llm_basics/simple_llm.py`
   - Then: `02_ai_agents/simple_agent.py`
   - Finally: `03_rag_system/rag_pipeline.py`

---

## 💻 Code Snippets to Try

### Send a Message to LLM
```python
from llm_ai_agent_rag.api_integration.openai_client import OpenAIClient

client = OpenAIClient()
response = client.simple_prompt("What is machine learning?")
print(response)
```

### Search Similar Documents
```python
from llm_ai_agent_rag.models.embedding_model import EmbeddingModel

model = EmbeddingModel()
docs = ["Data engineering processes data", "ML needs good data"]
results = model.semantic_search("data processing", docs)
print(results)
```

### Use an Agent
```python
from llm_ai_agent_rag.simple_agent import SimpleAgent

agent = SimpleAgent("Assistant")
agent.register_tool("calculator", lambda q: f"Calculated: {q}")
result = agent.execute("Calculate 5+3")
print(result)
```

---

## 📚 Project Structure at a Glance

```
llm_ai_agent_rag/
├── 01_llm_basics/      → Learn LLM APIs and prompting
├── 02_ai_agents/       → Build autonomous agents
├── 03_rag_system/      → Create RAG pipelines
├── 04_interview_prep/  → Interview questions & answers
├── README.md           → Full documentation
├── .env                → Your API keys
└── requirements.txt    → Install: pip install -r requirements.txt
```

---

## 🎯 Learning Path (5 Weeks)

**Week 1: LLM Basics**
- Study `01_llm_basics/simple_llm.py`
- Learn about prompts and models
- Practice with OpenAI API

**Week 2: Embeddings**
- Study `models/embedding_model.py`
- Learn semantic search
- Experiment with similarity

**Week 3: RAG**
- Study `03_rag_system/rag_pipeline.py`
- Build your first RAG system
- Test with sample documents

**Week 4: Agents**
- Study `02_ai_agents/simple_agent.py`
- Build multi-tool agents
- Implement memory

**Week 5: Interview**
- Review `04_interview_prep/interview_questions.md`
- Practice coding problems
- Study system design

---

## 🔑 Key Files to Know

| File | What It Does |
|------|-------------|
| `simple_llm.py` | Talk to GPT-3.5/GPT-4 |
| `embedding_model.py` | Convert text to vectors |
| `rag_pipeline.py` | Retrieve + Generate answers |
| `simple_agent.py` | Build autonomous agents |
| `interview_questions.md` | Ace your interviews |

---

## ✨ Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Verify API key in `.env` file
3. ✅ Run example: `python 01_llm_basics/simple_llm.py`
4. ✅ Read documentation in folder files
5. ✅ Modify and experiment with code
6. ✅ Build your own projects

---

## 🆘 Quick Help

**API Key not working?**
```bash
# Check .env file
nano .env

# Should have:
# OPENAI_API_KEY=sk-proj-your-actual-key
```

**Module not found?**
```bash
pip install openai anthropic sentence-transformers python-dotenv
```

**Want to learn more?**
- Read: `README.md`
- Check: `04_interview_prep/interview_questions.md`
- Run: All Python files with examples

---

## 🎓 You're Ready!

Your complete LLM/AI Agent/RAG learning environment is set up.

**Start now:** `python 01_llm_basics/simple_llm.py`

**Have fun! 🚀**
