"""
OpenAI API Integration
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")


class OpenAIClient:
    """Wrapper for OpenAI API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables"
            )
        self.client = OpenAI(api_key=self.api_key)

    def chat_completion(self, messages: list,
                        model: str = "gpt-3.5-turbo", **kwargs) -> str:
        """Send chat completion request"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def simple_prompt(self, prompt: str,
                      model: str = "gpt-3.5-turbo") -> str:
        """Simple single-turn prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, model=model)

    def chat_with_system(self, system_prompt: str, user_message: str,
                         model: str = "gpt-3.5-turbo") -> str:
        """Chat with system context"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.chat_completion(messages, model=model)

    def multi_turn_chat(self, messages: list,
                        model: str = "gpt-3.5-turbo") -> str:
        """Multi-turn conversation"""
        return self.chat_completion(messages, model=model)


if __name__ == "__main__":
    # Example usage
    try:
        client = OpenAIClient()

        # Simple prompt
        response = client.simple_prompt("What is data engineering in 2 sentences?")
        print("Response:", response)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
