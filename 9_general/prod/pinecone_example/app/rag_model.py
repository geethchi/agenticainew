from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("faq-embeddings")

def get_rag_response(query):

    # Create embedding
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    documents = [m["metadata"]["text"] for m in results["matches"]]

    prompt = f"Query: {query}\n\nContext:\n" + "\n".join(documents) + "\n\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    ).choices[0].message.content

    return response
