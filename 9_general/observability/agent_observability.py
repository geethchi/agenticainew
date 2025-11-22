from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os, requests, logging, time

from langgraph.graph import StateGraph, END
from typing import TypedDict
from langsmith import traceable

# --------------------------
# Logging Setup
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("news-agent-logger")

# --------------------------
# Shared State
# --------------------------
class AgentState(TypedDict):
    topic: str
    headlines: str
    summary: str
    sentiment: str
    final_report: str
    sentiment_label: str

# --------------------------
# Setup
# --------------------------
load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
news_key = os.getenv("NEWS_API_KEY")

# --------------------------
# Agent Functions (with observability)
# --------------------------

@traceable(name="fetch_news")
def fetch_news(state: AgentState) -> AgentState:
    logger.info(f"[fetch_news] topic={state['topic']}")
    url = f"https://newsapi.org/v2/everything?q={state['topic']}&apiKey={news_key}&pageSize=5"

    t0 = time.time()
    try:
        r = requests.get(url)
        articles = [a["title"] for a in r.json().get("articles", [])]
        headlines = "\n".join(articles)
        logger.info(f"[fetch_news] articles_found={len(articles)}")
    except Exception as e:
        logger.error(f"[fetch_news] error={e}")
        headlines = f"Error fetching news: {e}"

    logger.info(f"[fetch_news] duration={time.time()-t0:.2f}s")
    return {"headlines": headlines}


@traceable(name="summarize_news")
def summarize_news(state: AgentState) -> AgentState:
    logger.info("[summarize_news] summarizing headlines")
    prompt = f"Summarize these headlines about {state['topic']}:\n{state['headlines']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    logger.info("[summarize_news] summary_length=%d" % len(resp.content))
    return {"summary": resp.content}


@traceable(name="analyze_sentiment")
def analyze_sentiment(state: AgentState) -> AgentState:
    logger.info("[analyze_sentiment] analyzing sentiment")
    prompt = f"Is this summary overall positive or negative?\n\n{state['summary']}"

    resp = llm.invoke([HumanMessage(content=prompt)])
    sentiment = resp.content.strip().lower()
    label = "positive" if "positive" in sentiment else "negative"

    logger.info(f"[analyze_sentiment] sentiment={sentiment} label={label}")
    return {"sentiment": sentiment, "sentiment_label": label}


@traceable(name="investor_summary")
def investor_summary(state: AgentState) -> AgentState:
    logger.info("[investor_summary] generating investor report")
    prompt = f"Write an investor-focused insight based on:\n{state['summary']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}


@traceable(name="public_summary")
def public_summary(state: AgentState) -> AgentState:
    logger.info("[public_summary] generating public report")
    prompt = f"Write a short public news digest based on:\n{state['summary']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}

# --------------------------
# Build Agent Graph
# --------------------------
workflow = StateGraph(AgentState)

workflow.add_node("fetch_news", fetch_news)
workflow.add_node("summarize_news", summarize_news)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("investor_summary", investor_summary)
workflow.add_node("public_summary", public_summary)

workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "summarize_news")
workflow.add_edge("summarize_news", "analyze_sentiment")

workflow.add_conditional_edges(
    "analyze_sentiment",
    lambda state: "investor_summary" if state["sentiment_label"] == "positive" else "public_summary"
)

workflow.add_edge("investor_summary", END)
workflow.add_edge("public_summary", END)

app = workflow.compile()

# --------------------------
# Agent Runner (CLI)
# --------------------------
def run_agent(topic: str):
    logger.info(f"[run_agent] topic={topic}")

    initial = {
        "topic": topic,
        "headlines": "",
        "summary": "",
        "sentiment": "",
        "final_report": "",
        "sentiment_label": ""
    }

    result = app.invoke(initial)

    print("\n=== HEADLINES ===")
    print(result["headlines"])

    print("\n=== SUMMARY ===")
    print(result["summary"])

    print("\n=== SENTIMENT ===")
    print(f"{result['sentiment']} ({result['sentiment_label']})")

    print("\n=== FINAL REPORT ===")
    print(result["final_report"])
    print("\n")


# --------------------------
# MAIN (CLI Input)
# --------------------------
if __name__ == "__main__":
    print("AI News Agent (with Observability)")
    print("------------------------------------")
    while True:
        topic = input("\nEnter a topic (or type 'exit'): ").strip()
        if topic.lower() == "exit":
            print("Goodbye!")
            break
        run_agent(topic)
