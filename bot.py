import os
import logging
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)

def log_step(step_type: str, content: str):
    print(f"\n[{step_type}]\n{content}")
    logging.info(f"\n[{step_type}]\n{content}")

@tool
def requirement_structure_tool(requirement: str) -> dict:
  
    words = requirement.split()
    return {
        "analyzed_length": len(words),
        "status": "Requirement Received & Parsed",
    }

@tool
def generic_test_generator(action: str, expected_outcome: str) -> List[str]:
  
    return [
        f"POSITIVE: Verify user can '{action}' and see '{expected_outcome}'.",
        f"NEGATIVE: Verify '{action}' with empty data does NOT show '{expected_outcome}'.",
        f"BOUNDARY: Verify '{action}' with max character limit handles gracefully.",
        f"SECURITY: Verify '{action}' is protected against common vulnerabilities.",
    ]

@tool
def report_formatter(test_cases: List[str]) -> str:
   
    report = "--- QA AUTOMATION REPORT ---\n"
    for i, tc in enumerate(test_cases, 1):
        report += f"TC_{i:03d}: {tc}\n"
    report += "----------------------------"
    return report

TOOLS = [
    requirement_structure_tool,
    generic_test_generator,
    report_formatter,
]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)

class AgentState(TypedDict):
    messages: List[BaseMessage]

SYSTEM_PROMPT = """
You are a QA Assistant.
1. You have tools to analyze requirements, generate tests, and format reports.
2. DO NOT use tools if the user is just greeting you or asking general questions.
3. DECIDE which tool to use based on the specific request.
4. If no tool is needed, just reply with text.
"""

def llm_node(state: AgentState):
    messages = state["messages"]
    
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)

    if isinstance(response, AIMessage) and response.content:
        log_step("THOUGHT", response.content)

    if isinstance(response, AIMessage) and response.tool_calls:
        for call in response.tool_calls:
            log_step("ACTION", f"Tool: {call['name']} | Args: {call['args']}")

    return {"messages": [response]}

tool_node = ToolNode(TOOLS)

def route(state: AgentState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    route,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "llm")

app = graph.compile()

def run_bot():
    print("QEA Agent")
    
    try:
        user_input = input("Enter your input: ").strip()
        if not user_input:
            return

        log_step("INPUT", user_input)

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )

        print("\nFINAL OUTPUT:")
        print(result["messages"][-1].content)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_bot()
