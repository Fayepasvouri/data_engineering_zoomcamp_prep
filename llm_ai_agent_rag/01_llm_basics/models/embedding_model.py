"""
Embedding Models for RAG and Semantic Search
"""
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install sentence-transformers: pip install sentence-transformers")


class EmbeddingModel:
    """Wrapper for sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Popular models:
        - all-MiniLM-L6-v2 (small, fast)
        - all-mpnet-base-v2 (larger, better quality)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list) -> np.ndarray:
        """Convert texts to embeddings"""
        return self.model.encode(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def semantic_search(self, query: str, documents: list,
                        top_k: int = 5) -> list:
        """Find most similar documents to query"""
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode(documents)

        similarities = np.dot(doc_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [documents[i] for i in top_indices]


if __name__ == "__main__":
    # Test embedding model
    model = EmbeddingModel()

    # Test similarity
    text1 = "Machine learning is powerful"
    text2 = "AI is amazing"
    similarity = model.similarity(text1, text2)
    print(f"Similarity between texts: {similarity:.4f}")

    # Test semantic search
    query = "data engineering"
    documents = [
        "ETL pipelines are used for data processing",
        "Machine learning models need data",
        "Data warehouses store large amounts of data",
        "Python is a programming language",
    ]
    results = model.semantic_search(query, documents, top_k=2)
    print(f"\nTop results for '{query}':")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")
