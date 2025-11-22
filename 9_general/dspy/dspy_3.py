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

from typing import Literal

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
# result = classify(sentence="This book was super fun to read, though not the last chapter.")
result = classify(sentence="The product quality was average. Packing was poor, though.")

print(result)
