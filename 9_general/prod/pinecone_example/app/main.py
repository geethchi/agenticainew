from flask import Flask, request, jsonify
from rag_model import get_rag_response
from cache import get_cached_response, set_cached_response


app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "RAG API is running"})

@app.route("/query/", methods=["POST"])
def query_rag():
    """
    Endpoint to handle RAG queries.
    Checks the cache first before generating a response.
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query is required"}), 400

    query = data["query"]

    # Check cache first
    cached_response = get_cached_response(query)
    if cached_response:
        return jsonify({"response": cached_response, "source": "cache"})

    # Get fresh RAG response
    response = get_rag_response(query)
    set_cached_response(query, response)

    return jsonify({"response": response, "source": "RAG"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
