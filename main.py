from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
import json
from langgraph.graph import StateGraph, END

# Define the state schema using TypedDict
class State(TypedDict):
    idea: str
    validated_idea: str
    deepcrawler_instructions: str

# Initialize LLaMA 2 model via Ollama
llm = OllamaLLM(
    model="llama2:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
    timeout=60
)

# VerifAgent: Validates the business idea
verif_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a validation expert in the field of Artificial Intelligence and Data Science. "
     "Validate the business idea for its relevance, feasibility, and value within AI or Data Science. "
     "Summarize its potential, challenges, and opportunities in under 200 words. "
     "If the idea is not related to AI or Data Science, respond with: 'This idea is not related to AI or Data Science.'"),
    ("human", "{idea}")
])
verif_agent = verif_prompt | llm

# MetaAgent: Generates instructions for the DeepCrawler Agent
meta_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the DeepCrawler Agent embedded within an agentic system that analyzes the AI and Data industry using Porter's Five Forces framework. "
     "Your task is to autonomously plan and execute a deep market and ecosystem crawl using tools like Tavily, Gemini, and MCP APIs. "
     "The pipeline should adapt based on the strategic context provided.\n\n"
     "Instructions:\n"
     "1. Based on the selected **Porter Force** below, formulate a data acquisition and crawling strategy.\n"
     "2. Use your understanding of the AI and Data industry to customize your queries, APIs, and exploration depth.\n"
     "3. Return structured results including URLs, summaries, sentiment analysis, and relevance scores.\n\n"
     "Inputs:\n"
     "- **Porter Force**: {{porter_force}}\n"
     "- **Project Idea / Business Context**: {{project_idea}}\n\n"
     "Output Format:\n"
     "- List of primary data sources used (Tavily, APIs, Gemini responses)\n"
     "- Top 10 relevant entities (e.g., companies, products, technologies, patents)\n"
     "- Link and metadata for each entity (e.g., title, summary, tags, confidence score)\n"
     "- Observations tied back to the force (e.g., how buyer power is increasing due to XYZ)\n"
     "- Confidence score for each observation\n"
     "- Suggestions for next best crawling targets (iterative learning)\n\n"
     "Behavior by Force:\n"
     "- *Buyer Power*: Identify key buyer segments, emerging use cases, price sensitivity, adoption trends.\n"
     "- *Supplier Power*: Scrape vendor ecosystems, toolchains, dependency graphs, and monopolies.\n"
     "- *New Entrants*: Detect startups, product launches, hackathons, funding rounds.\n"
     "- *Substitutes*: Explore parallel industries, alternative tooling, competitive innovations.\n"
     "- *Rivalry*: Benchmark players, marketing strategies, talent wars, or acquisition trends.\n\n"
     "Constraints:\n"
     "- You must respect API limits and respond in under 60 seconds.\n"
     "- Filter low-relevance results using cosine similarity or Gemini-generated summary validation.\n"
     "- Prioritize freshness (past 6–12 months) and regionally relevant data.\n\n"
     "Goal:\n"
     "Provide rich, dynamic, and actionable insights that help strategic analysts understand the force-specific landscape surrounding the project idea in the AI/Data space.\n\n"
     "Begin execution based on:\n"
     "- {{porter_force}}\n"
     "- {{project_idea}}"),
    ("human", 
     "Generate comprehensive DeepCrawler Agent instructions for analyzing this business idea across all Porter's Five Forces:\n"
     "Business Idea: {validated_idea}\n\n"
     "Create detailed instructions that will guide the DeepCrawler Agent to systematically analyze each of the five forces: "
     "Buyer Power, Supplier Power, Threat of New Entrants, Threat of Substitutes, and Competitive Rivalry.")
])

meta_agent = meta_prompt | llm

# Convert agent outputs to dictionary state
def extract_validated_idea(response):
    return {"validated_idea": response.strip()}

def extract_deepcrawler_instructions(response):
    return {"deepcrawler_instructions": response.strip()}

# Initialize LangGraph
graph = StateGraph(State)

graph.add_node("verif_agent", verif_agent | RunnableLambda(extract_validated_idea))
graph.add_node("meta_agent", meta_agent | RunnableLambda(extract_deepcrawler_instructions))

graph.set_entry_point("verif_agent")
graph.add_edge("verif_agent", "meta_agent")
graph.add_edge("meta_agent", END)

workflow = graph.compile()

# Get user input for the business idea
idea = input("Enter your business idea: ").strip()
if not idea:
    idea = "AI-powered optimization platform for car rental fleet usage and predictive maintenance"

input_data = {"idea": idea}
result = workflow.invoke(input_data)

print("Generated DeepCrawler Agent Instructions:")
print("="*50)
print(result["deepcrawler_instructions"])
print("\n" + "="*50)
print("Validated Business Idea:")
print(result["validated_idea"])