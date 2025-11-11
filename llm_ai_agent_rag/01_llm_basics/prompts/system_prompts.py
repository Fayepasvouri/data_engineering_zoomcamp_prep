"""
Prompt Templates for LLM interactions
"""

SYSTEM_PROMPTS = {
    "data_engineer": """You are an expert data engineer. 
Help users with data pipeline design, ETL processes, and data architecture questions.
Provide practical, production-ready solutions.""",
    
    "code_reviewer": """You are an expert code reviewer.
Analyze code for bugs, performance issues, and best practices.
Provide constructive feedback with examples.""",
    
    "tutor": """You are a helpful tutor.
Explain complex concepts in simple terms.
Use examples and analogies to help understanding.""",
    
    "interview_coach": """You are an expert interview coach.
Help candidates prepare for technical interviews.
Provide tips, example answers, and constructive feedback.""",
}

INTERVIEW_PROMPTS = {
    "system_design": """Design a system for: {question}
Consider: scalability, latency, fault tolerance.
Provide architecture diagram description.""",
    
    "coding": """Solve this coding problem: {question}
Provide: approach, code, complexity analysis.""",
    
    "behavioral": """Answer this behavioral question: {question}
Use STAR method: Situation, Task, Action, Result.""",
}

def get_prompt(category: str, prompt_type: str, **kwargs) -> str:
    """Get a formatted prompt"""
    if category == "system":
        return SYSTEM_PROMPTS.get(prompt_type, "")
    elif category == "interview":
        template = INTERVIEW_PROMPTS.get(prompt_type, "")
        return template.format(**kwargs)
    return ""

if __name__ == "__main__":
    # Test the prompts
    print("Data Engineer Prompt:")
    print(get_prompt("system", "data_engineer"))
    print("\n" + "="*50 + "\n")
    print("Interview Question:")
    print(get_prompt("interview", "coding", question="Reverse a linked list"))
