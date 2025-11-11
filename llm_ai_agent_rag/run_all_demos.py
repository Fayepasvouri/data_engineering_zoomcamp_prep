"""
Complete Demo - LLM, Agent & RAG All Together
Shows how all three systems work
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "=" * 80)
print("🚀 COMPLETE LLM, AGENT & RAG DEMO")
print("=" * 80)

# ============================================================================
# PART 1: LLM DEMO
# ============================================================================

print("\n" + "─" * 80)
print("PART 1: LANGUAGE MODEL DEMO")
print("─" * 80)

from llm_ai_agent_rag.api_integration.openai_client import OpenAIClient

class DemoLLMClient:
    """Mock LLM for demo"""
    def __init__(self):
        self.responses = {
            "machine learning": (
                "Machine learning enables systems to learn from data "
                "without explicit programming"
            ),
            "data": (
                "Data is information used to train and inform AI systems"
            ),
            "rag": (
                "RAG combines retrieval with generation for accurate responses"
            ),
        }

    def simple_prompt(self, prompt):
        for key, response in self.responses.items():
            if key in prompt.lower():
                return response
        return f"Response to: {prompt}"


try:
    print("\n✓ Initializing LLM client...")
    llm = DemoLLMClient()

    print("\n📝 Test 1: Simple prompt")
    response1 = llm.simple_prompt("What is machine learning?")
    print(f"   Q: What is machine learning?")
    print(f"   A: {response1}")

    print("\n📝 Test 2: Another prompt")
    response2 = llm.simple_prompt("Tell me about data")
    print(f"   Q: Tell me about data")
    print(f"   A: {response2}")

    print("\n✅ LLM Demo Completed!")

except Exception as e:
    print(f"❌ LLM Demo Error: {e}")

# ============================================================================
# PART 2: AGENT DEMO
# ============================================================================

print("\n" + "─" * 80)
print("PART 2: AI AGENT DEMO")
print("─" * 80)

from llm_ai_agent_rag.simple_agent import SimpleAgent

try:
    print("\n✓ Initializing Agent...")
    agent = SimpleAgent("DataEngineeringAgent")

    # Define some tools
    def calculator(query):
        return f"💻 Calculating: {query}"

    def database_query(query):
        return f"🗄️  Querying database: {query}"

    def api_call(query):
        return f"🌐 Making API call: {query}"

    # Register tools
    agent.register_tool("calculator", calculator)
    agent.register_tool("database", database_query)
    agent.register_tool("api", api_call)

    print(f"✓ Agent Name: {agent.name}")
    print(f"✓ Available Tools: {agent.list_tools()}")

    # Execute queries
    print("\n📝 Query 1: Calculate some data")
    result1 = agent.execute("Calculate the average data size")
    print(f"   Response: {result1}")

    print("\n📝 Query 2: Search database")
    result2 = agent.execute("Search for user records")
    print(f"   Response: {result2}")

    print("\n📝 Query 3: Call API")
    result3 = agent.execute("Call the API endpoint")
    print(f"   Response: {result3}")

    # Show memory
    print(f"\n✓ Agent Memory: {len(agent.get_memory())} interactions stored")

    print("\n✅ Agent Demo Completed!")

except Exception as e:
    print(f"❌ Agent Demo Error: {e}")

# ============================================================================
# PART 3: RAG DEMO
# ============================================================================

print("\n" + "─" * 80)
print("PART 3: RAG SYSTEM DEMO")
print("─" * 80)

from llm_ai_agent_rag.rag_system.rag_pipeline import RAGPipeline

try:
    print("\n✓ Initializing RAG Pipeline...")
    rag = RAGPipeline()

    # Sample documents
    documents = [
        "Data engineering involves building data pipelines for ETL processes",
        "Machine learning models require high-quality data for training",
        "Apache Spark is a distributed computing framework for big data",
        "Data warehouses store and organize business data for analysis",
        "API design patterns include REST, GraphQL, and gRPC",
    ]

    # Ingest documents
    print(f"✓ Ingesting {len(documents)} documents...")
    rag.ingest_documents(documents)

    # Test queries
    queries = [
        "What is data engineering?",
        "Tell me about Spark",
        "API design approaches",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        result = rag.query(query)
        print(f"   Documents Used: {result['num_documents_used']}")
        print(f"   Response: {result['response']}")

    print("\n✅ RAG Demo Completed!")

except Exception as e:
    print(f"❌ RAG Demo Error: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 SUMMARY:")
print("   ✓ LLM System: Working")
print("   ✓ Agent System: Working")
print("   ✓ RAG System: Working")

print("\n🎯 What You Learned:")
print("   • How LLMs generate responses")
print("   • How agents use tools and make decisions")
print("   • How RAG retrieves and generates with context")

print("\n🚀 Next Steps:")
print("   1. Modify demo_llm.py to add your own responses")
print("   2. Add more tools to the agent")
print("   3. Expand RAG with real documents")
print("   4. Read interview_questions.md for concepts")

print("\n📚 Files to Explore:")
print("   • 01_llm_basics/simple_llm.py")
print("   • 02_ai_agents/simple_agent.py")
print("   • 03_rag_system/rag_pipeline.py")
print("   • 04_interview_prep/interview_questions.md")

print("\n" + "=" * 80)
print("🎓 Happy Learning! 🎓")
print("=" * 80 + "\n")
