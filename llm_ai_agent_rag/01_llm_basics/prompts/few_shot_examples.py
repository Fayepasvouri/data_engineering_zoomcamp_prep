"""
Few-shot prompt examples for better LLM responses
"""

DATA_ENGINEERING_EXAMPLES = [
    {
        "input": "Design a real-time data pipeline",
        "output": """Architecture:
1. Data Source → Kafka (message queue)
2. Stream Processor (Spark/Flink)
3. Data Warehouse (Snowflake/BigQuery)
4. Analytics Layer (BI tools)

Benefits: Low latency, scalability, fault tolerance"""
    },
    {
        "input": "How to handle late-arriving data?",
        "output": """Solutions:
1. Watermarking: Define acceptable lateness window
2. Session windows: Group events by session ID
3. Side outputs: Capture late data separately
4. Re-process: Replay from source if needed"""
    }
]

CODE_REVIEW_EXAMPLES = [
    {
        "code": "for i in range(len(list)): print(list[i])",
        "feedback": "Use pythonic iteration: for item in list: print(item)"
    },
    {
        "code": "if x == True: pass",
        "feedback": "Simplify to: if x: pass"
    }
]


def get_few_shot_prompt(domain: str, num_examples: int = 2) -> str:
    """Generate few-shot examples for a domain"""
    if domain == "data_engineering":
        examples = DATA_ENGINEERING_EXAMPLES[:num_examples]
    elif domain == "code_review":
        examples = CODE_REVIEW_EXAMPLES[:num_examples]
    else:
        examples = []

    prompt = "Learn from these examples:\n\n"
    for ex in examples:
        prompt += f"Input: {ex.get('input', ex.get('code'))}\n"
        prompt += f"Output: {ex.get('output', ex.get('feedback'))}\n\n"

    return prompt


if __name__ == "__main__":
    # Test few-shot prompts
    print("Data Engineering Examples:")
    print(get_few_shot_prompt("data_engineering"))
    print("\n" + "="*50 + "\n")
    print("Code Review Examples:")
    print(get_few_shot_prompt("code_review"))
