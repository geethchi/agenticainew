import dspy
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 1. Create LM
lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. Configure DSPy to use this LM
dspy.configure(lm=lm)

# 3. Run a Chain-of-Thought program
math = dspy.ChainOfThought("question -> answer: float")

result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

print(result)
