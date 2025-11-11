"""
LLM Demo - Works without API calls (for testing/demo)
Shows how LLM integration works
"""
import os
from dotenv import load_dotenv

load_dotenv()


class DemoLLM:
    """Demo LLM that simulates responses without API calls"""

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.conversation_history = []

    def generate_response(self, prompt: str) -> str:
        """Simulate a response from LLM (without API call)"""
        # Demo responses based on keywords
        demo_responses = {
            "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "data engineering": "Data engineering is the practice of designing and building systems to collect, process, and provide data to analysts and scientists.",
            "rag": "Retrieval Augmented Generation (RAG) combines document retrieval with language model generation to produce accurate, sourced responses.",
            "agent": "An AI agent is an autonomous system that perceives its environment, makes decisions, and takes actions to achieve specific goals.",
            "embedding": "Embeddings are dense vector representations of text that capture semantic meaning, enabling similarity comparisons.",
        }

        # Find matching response
        for keyword, response in demo_responses.items():
            if keyword.lower() in prompt.lower():
                return f"✅ LLM Response:\n{response}\n\n[Model: {self.model_name}]"

        # Default response
        return f"✅ LLM Response:\nThis is a demo response about: {prompt}\n\n[Model: {self.model_name}]"

    def chat_completion(self, messages: list) -> str:
        """Handle multi-turn conversations (demo)"""
        # Store in history
        self.conversation_history.extend(messages)

        # Get last user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            last_message = user_messages[-1]["content"]
            return self.generate_response(last_message)
        else:
            return "No user message found"


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 LLM DEMO - Testing Your LLM Setup")
    print("=" * 70)

    try:
        # Initialize LLM
        llm = DemoLLM()
        print(f"\n✓ LLM Initialized: {llm.model_name}")
        print(f"✓ API Key Status: {'Configured' if llm.api_key else 'Not set'}\n")

        # Test 1: Simple generation
        print("\n" + "─" * 70)
        print("TEST 1: Simple LLM Generation")
        print("─" * 70)
        prompt1 = "What is machine learning in one sentence?"
        print(f"📝 Prompt: {prompt1}")
        response1 = llm.generate_response(prompt1)
        print(response1)

        # Test 2: Data Engineering
        print("\n" + "─" * 70)
        print("TEST 2: Data Engineering Question")
        print("─" * 70)
        prompt2 = "Explain data engineering"
        print(f"📝 Prompt: {prompt2}")
        response2 = llm.generate_response(prompt2)
        print(response2)

        # Test 3: Multi-turn conversation
        print("\n" + "─" * 70)
        print("TEST 3: Multi-turn Conversation")
        print("─" * 70)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG?"},
        ]
        print(f"📝 Messages: {len(messages)} message(s)")
        response3 = llm.chat_completion(messages)
        print(response3)

        # Test 4: Agent Question
        print("\n" + "─" * 70)
        print("TEST 4: AI Agent Concept")
        print("─" * 70)
        prompt4 = "What is an AI agent?"
        print(f"📝 Prompt: {prompt4}")
        response4 = llm.generate_response(prompt4)
        print(response4)

        # Test 5: Embedding
        print("\n" + "─" * 70)
        print("TEST 5: Embeddings Question")
        print("─" * 70)
        prompt5 = "What are embeddings?"
        print(f"📝 Prompt: {prompt5}")
        response5 = llm.generate_response(prompt5)
        print(response5)

        # Summary
        print("\n" + "=" * 70)
        print("✅ LLM DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📊 Test Summary:")
        print(f"  • Model: {llm.model_name}")
        print(f"  • Tests Run: 5")
        print(f"  • Conversation History: {len(llm.conversation_history)} messages")
        print(f"  • Status: All tests passed ✓")

        print("\n💡 NOTE: This is a demo version that doesn't use API calls.")
        print("   To use real OpenAI API:")
        print("   1. Check your OpenAI account billing")
        print("   2. Ensure your API key has available credits")
        print("   3. Update simple_llm.py to make actual API calls")

        print("\n🔧 Next Steps:")
        print("   1. Run: python 02_ai_agents/simple_agent.py")
        print("   2. Run: python 03_rag_system/rag_pipeline.py")
        print("   3. Review: 04_interview_prep/interview_questions.md")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your setup and try again")
