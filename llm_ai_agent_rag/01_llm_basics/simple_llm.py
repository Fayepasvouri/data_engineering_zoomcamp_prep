"""
Simple LLM Integration - Learn how to use LLM APIs
"""
import os
from dotenv import load_dotenv

load_dotenv()


class SimpleLLM:
    """Basic LLM wrapper for learning"""

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set in .env file"
            )

    def generate_response(self, prompt: str) -> str:
        """Generate a simple response from the LLM"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "Please install openai: pip install openai"

    def chat_completion(self, messages: list) -> str:
        """Handle multi-turn conversations"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except ImportError:
            return "Please install openai: pip install openai"


if __name__ == "__main__":
    try:
        llm = SimpleLLM()

        # Test simple generation
        response = llm.generate_response(
            "What is machine learning in one sentence?"
        )
        print("LLM Response:")
        print(response)

        # Test multi-turn
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is data engineering?"},
        ]
        response2 = llm.chat_completion(messages)
        print("\nMulti-turn Response:")
        print(response2)

    except ValueError as e:
        print(f"Error: {e}")
