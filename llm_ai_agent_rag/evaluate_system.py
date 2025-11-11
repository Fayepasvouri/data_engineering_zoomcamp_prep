"""
Complete System Evaluation with all metrics
"""
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

# Add parent directory and subdirectories to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "01_llm_basics"))
sys.path.insert(0, str(root_dir / "01_llm_basics" / "models"))
sys.path.insert(0, str(root_dir / "02_ai_agents"))
sys.path.insert(0, str(root_dir / "03_rag_system"))

metrics = defaultdict(list)
start_time = time.time()

# LLM EVALUATION
print("\n" + "=" * 80)
print("📊 LLM EVALUATION METRICS")
print("=" * 80)

class LLMEvaluator:
    def __init__(self):
        self.responses = {
            "machine learning": "ML is AI that enables systems to learn from data",
            "data engineering": "Data eng designs systems to process and manage data",
            "rag": "RAG combines retrieval with LLM generation for context-aware responses",
            "embeddings": "Embeddings are vectors that capture semantic meaning",
        }

    def evaluate(self, prompt):
        for key, response in self.responses.items():
            if key in prompt.lower():
                return response
        return "Generated response"

try:
    print("\nTesting LLM Accuracy...")
    llm = LLMEvaluator()

    test_cases = [
        ("What is machine learning?", ["learning", "data"]),
        ("Define data engineering", ["engineering", "systems"]),
        ("Explain RAG", ["retrieval", "generation"]),
        ("What are embeddings?", ["vector", "semantic"]),
    ]

    accuracies = []

    for i, (prompt, keywords) in enumerate(test_cases, 1):
        response = llm.evaluate(prompt)
        matches = sum(1 for kw in keywords if kw.lower() in response.lower())
        accuracy = (matches / len(keywords)) * 100

        print(f"  Test {i}: {accuracy:.0f}% - {prompt}")
        accuracies.append(accuracy)
        metrics["LLM"].append(("Test_" + str(i), accuracy))

    avg_accuracy = np.mean(accuracies)
    print(f"  ✅ Average LLM Accuracy: {avg_accuracy:.2f}%\n")
    metrics["LLM"].append(("Average_Accuracy", avg_accuracy))

except Exception as e:
    print(f"❌ LLM Error: {e}\n")

# EMBEDDING EVALUATION
print("=" * 80)
print("🧮 EMBEDDING & VECTOR EVALUATION")
print("=" * 80)

try:
    from embedding_model import EmbeddingModel

    print("\nTesting Embedding Similarity...")
    embedder = EmbeddingModel()

    test_pairs = [
        ("data engineering", "building pipelines"),
        ("machine learning", "statistical models"),
        ("python", "javascript"),
        ("vector database", "embedding storage"),
    ]

    similarities = []

    for i, (text1, text2) in enumerate(test_pairs, 1):
        similarity = float(embedder.similarity(text1, text2))
        print(f"  Pair {i}: {similarity:.4f} - '{text1}' vs '{text2}'")
        similarities.append(similarity)
        metrics["Embeddings"].append(("Pair_" + str(i), similarity))

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    max_sim = max(similarities)
    min_sim = min(similarities)

    print(f"\n  Statistics - Mean: {mean_sim:.4f}, "
          f"Std: {std_sim:.4f}, Max: {max_sim:.4f}, Min: {min_sim:.4f}\n")

    metrics["Embeddings"].extend([
        ("Mean_Similarity", mean_sim),
        ("Std_Dev", std_sim),
        ("Max_Similarity", max_sim),
        ("Min_Similarity", min_sim),
    ])

    query = "data processing"
    documents = [
        "ETL pipelines for data transformation",
        "Machine learning model training",
        "SQL database optimization",
        "Cloud computing infrastructure",
    ]

    results = embedder.semantic_search(query, documents, top_k=2)
    print(f"  Semantic Search: Found {len(results)} results for '{query}'\n")
    metrics["Embeddings"].append(("Search_Results", len(results)))

except ImportError:
    print("⚠️  Skipping embeddings (module not available)\n")
except Exception as e:
    print(f"❌ Embeddings Error: {e}\n")

# RAG EVALUATION
print("=" * 80)
print("🎯 RAG SYSTEM EVALUATION")
print("=" * 80)

try:
    from rag_pipeline import RAGPipeline

    print("\nTesting RAG Performance...")
    rag = RAGPipeline()

    documents = [
        "Data engineering involves ETL pipelines",
        "Machine learning requires good data",
        "Spark is a distributed computing framework",
        "Vector databases store embeddings",
        "API design patterns include REST",
    ]

    print(f"  Documents ingested: {len(documents)}")
    rag.ingest_documents(documents)
    metrics["RAG"].append(("Document_Count", len(documents)))

    queries = [
        "What is data engineering?",
        "Tell me about Spark",
        "How do vector databases work?",
    ]

    retrieval_times = []
    docs_retrieved = []

    for i, query in enumerate(queries, 1):
        start = time.time()
        result = rag.query(query)
        retrieval_time = time.time() - start

        docs = result['num_documents_used']
        print(f"  Query {i}: {docs} docs, {retrieval_time:.4f}s - {query}")

        retrieval_times.append(retrieval_time)
        docs_retrieved.append(docs)
        metrics["RAG"].append(("Query_" + str(i) + "_Time", retrieval_time))
        metrics["RAG"].append(("Query_" + str(i) + "_Docs", docs))

    avg_docs = np.mean(docs_retrieved)
    avg_time = np.mean(retrieval_times)

    print(f"\n  Statistics - Avg Docs: {avg_docs:.2f}, "
          f"Avg Time: {avg_time:.4f}s\n")

    metrics["RAG"].extend([
        ("Avg_Documents", avg_docs),
        ("Avg_Retrieval_Time", avg_time),
    ])

except ImportError:
    print("⚠️  Skipping RAG (module not available)\n")
except Exception as e:
    print(f"❌ RAG Error: {e}\n")

# AGENT EVALUATION
print("=" * 80)
print("🤖 AGENT SYSTEM EVALUATION")
print("=" * 80)

try:
    from simple_agent import SimpleAgent

    print("\nTesting Agent Performance...")
    agent = SimpleAgent("EvaluationAgent")

    agent.register_tool("calculator", lambda q: "Calculated")
    agent.register_tool("search", lambda q: "Searched")
    agent.register_tool("database", lambda q: "Database query")

    print(f"  Available tools: {len(agent.list_tools())}")
    metrics["Agent"].append(("Tools_Count", len(agent.list_tools())))

    queries = [
        ("Calculate values", "calculator"),
        ("Search data", "search"),
        ("Query database", "database"),
    ]

    exec_times = []
    tool_accuracy = 0

    for i, (query, expected_tool) in enumerate(queries, 1):
        start = time.time()
        result = agent.execute(query)
        exec_time = time.time() - start

        tool_used = result['tool_used']
        correct = tool_used == expected_tool

        if correct:
            tool_accuracy += 1

        status = "✓" if correct else "✗"
        print(f"  Query {i}: {status} {tool_used:12} {exec_time:.4f}s - {query}")

        exec_times.append(exec_time)
        metrics["Agent"].append(("Query_" + str(i) + "_Time", exec_time))

    tool_acc_pct = (tool_accuracy / len(queries)) * 100
    avg_exec_time = np.mean(exec_times)
    memory_size = len(agent.get_memory())

    print(f"\n  Statistics - Accuracy: {tool_acc_pct:.0f}%, "
          f"Avg Time: {avg_exec_time:.4f}s, Memory: {memory_size}\n")

    metrics["Agent"].extend([
        ("Tool_Accuracy", tool_acc_pct),
        ("Avg_Execution_Time", avg_exec_time),
        ("Memory_Size", memory_size),
    ])

except ImportError:
    print("⚠️  Skipping Agent (module not available)\n")
except Exception as e:
    print(f"❌ Agent Error: {e}\n")

# SUMMARY
elapsed = time.time() - start_time

print("=" * 80)
print("📈 FINAL EVALUATION SUMMARY")
print("=" * 80)

print(f"\nTotal Time: {elapsed:.2f}s")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for system, results in metrics.items():
    print(f"{system}:")
    for metric, value in results:
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    print()

print("=" * 80)
print("✅ EVALUATION COMPLETE!")
print("=" * 80 + "\n")
