import dspy
lm = dspy.LM("ollama_chat/llama3.2:latest", api_base="http://localhost:11434", api_key="")

# 2. Configure DSPy to use this LM
dspy.configure(lm=lm)

# 3. Run a Chain-of-Thought program
math = dspy.ChainOfThought("question -> answer: float")

result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

print(result)
