# pip install pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

# Clients
client = OpenAI()
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your index
index = pinecone_client.Index("faq-embeddings")

# FAQ dictionary
faq_database = {
    "What is your return policy?": "Our return policy allows customers to return products within 30 days of purchase...",
    "How do I track my order?": "You can track your order using the tracking number provided...",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and Apple Pay...",
    "Can I change or cancel my order after it’s been placed?": "Once an order has been placed, we cannot modify it...",
    "What are your shipping options?": "We offer standard, expedited, and overnight shipping...",
    "How do I reset my account password?": "To reset your password, click 'Forgot Password'...",
    "Do you ship internationally?": "Yes, we ship to select international destinations...",
    "What do I do if I receive a damaged or defective product?": "Contact support within 48 hours...",
    "How do I contact customer support?": "Email support@ourcompany.com or call 1-800-123-4567...",
    "Can I use multiple discount codes on a single order?": "No, only one discount code can be used...",
}

# Convert FAQs into upsertable format
vectors_to_upsert = []

for i, (question, answer) in enumerate(faq_database.items()):
    # Create embedding for question + answer
    text = question + "\n" + answer

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    vectors_to_upsert.append({
        "id": f"faq-{i}",
        "values": embedding,
        "metadata": {
            "question": question,
            "answer": answer
        }
    })

# Upsert
index.upsert(vectors=vectors_to_upsert, namespace="ns-1")

print(f"Uploaded {len(vectors_to_upsert)} FAQs to Pinecone.")
