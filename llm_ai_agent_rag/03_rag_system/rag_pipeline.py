"""
RAG (Retrieval Augmented Generation) Pipeline
Simple implementation for learning
"""
from typing import List, Dict
import numpy as np


class RAGPipeline:
    """Simple RAG Pipeline for learning"""

    def __init__(self, retriever=None, llm_client=None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.documents: List[str] = []
        self.embeddings = []

    def ingest_documents(self, docs: List[str]):
        """Load and index documents"""
        self.documents = docs
        print(f"Ingested {len(docs)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant documents"""
        if not self.documents:
            return []

        # Simple keyword matching (placeholder)
        query_words = query.lower().split()
        scores = []

        for doc in self.documents:
            score = sum(1 for word in query_words if word in doc.lower())
            scores.append(score)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

    def generate(self, query: str, context: List[str]) -> str:
        """Generate response with retrieved context"""
        context_str = "\n".join(context)
        prompt = f"""Based on the following context:
{context_str}

Answer this question: {query}
"""
        if self.llm_client:
            return self.llm_client.simple_prompt(prompt)
        else:
            return f"Generated response based on {len(context)} documents"

    def query(self, question: str) -> Dict:
        """Full RAG pipeline"""
        # Retrieve
        context = self.retrieve(question)

        # Generate
        response = self.generate(question, context)

        return {
            "question": question,
            "context_used": context,
            "response": response,
            "num_documents_used": len(context)
        }


if __name__ == "__main__":
    # Create RAG pipeline
    rag = RAGPipeline()

    # Sample documents
    sample_docs = [
        "Data engineering involves building pipelines to process data",
        "ETL stands for Extract, Transform, Load",
        "Machine learning requires clean data",
        "Apache Spark is a distributed computing framework",
        "Data warehouses store large amounts of structured data",
    ]

    # Ingest documents
    rag.ingest_documents(sample_docs)

    # Test queries
    queries = [
        "What is data engineering?",
        "What does ETL mean?",
        "Tell me about Spark",
    ]

    print("\n" + "="*60)
    print("RAG Pipeline Test")
    print("="*60 + "\n")

    for q in queries:
        result = rag.query(q)
        print(f"Question: {result['question']}")
        print(f"Docs used: {result['num_documents_used']}")
        print(f"Response: {result['response']}\n")
