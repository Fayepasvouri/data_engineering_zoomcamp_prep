"""
Anthropic Claude API Integration
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from anthropic import Anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")


class AnthropicClient:
    """Wrapper for Anthropic Claude API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables"
            )
        self.client = Anthropic(api_key=self.api_key)

    def send_message(self, message: str,
                     model: str = "claude-3-sonnet-20240229") -> str:
        """Send a message to Claude"""
        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text

    def multi_turn_conversation(
            self,
            messages: list,
            model: str = "claude-3-sonnet-20240229"
    ) -> str:
        """Multi-turn conversation with Claude"""
        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=messages
        )
        return response.content[0].text


if __name__ == "__main__":
    # Example usage
    try:
        client = AnthropicClient()

        # Simple message
        response = client.send_message(
            "What is RAG in machine learning?"
        )
        print("Claude Response:", response)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ANTHROPIC_API_KEY in your .env file")
