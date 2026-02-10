import os
import logging
from typing import List, TypedDict
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --------------------------------------------------
# 1. LOAD ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)

def log_step(step_type: str, content: str):
    output = f"\n[{step_type}]\n{content}"
    print(output)
    logging.info(output)

# --------------------------------------------------
# 3. TOOLS
# --------------------------------------------------

@tool
def requirement_structure_tool(requirement: str) -> dict:
    """
    Takes raw requirement text and confirms understanding.
    """
    words = requirement.split()
    return {
        "analyzed_length": len(words),
        "status": "Requirement Received & Parsed",
    }

@tool
def generic_test_generator(action: str, expected_outcome: str) -> List[str]:
    """
    Generates generic QA test cases.
    """
    return [
        f"POSITIVE: Verify user can '{action}' and see '{expected_outcome}'.",
        f"NEGATIVE: Verify '{action}' with empty data does NOT show '{expected_outcome}'.",
        f"BOUNDARY: Verify '{action}' with max character limit handles gracefully.",
        f"SECURITY: Verify '{action}' is protected against common vulnerabilities.",
    ]

@tool
def report_formatter(test_cases: List[str]) -> str:
    """
    Formats test cases into final report.
    """
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

# --------------------------------------------------
# 4. GEMINI LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0,
    convert_system_message_to_human=True,
)

llm_with_tools = llm.bind_tools(TOOLS)

# --------------------------------------------------
# 5. LANGGRAPH STATE
# --------------------------------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]

# --------------------------------------------------
# 6. GRAPH NODES
# --------------------------------------------------

def llm_node(state: AgentState):
    """
    LLM reasoning node
    """
    response = llm_with_tools.invoke(state["messages"])

    if isinstance(response, AIMessage) and response.content:
        log_step("THOUGHT", response.content)

    if isinstance(response, AIMessage) and response.tool_calls:
        for call in response.tool_calls:
            log_step(
                "ACTION",
                f"Tool: {call['name']} | Args: {call['args']}",
            )

    return {"messages": state["messages"] + [response]}

tool_node = ToolNode(TOOLS)

def route(state: AgentState):
    """
    Decide whether to call tools or end.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END

# --------------------------------------------------
# 7. BUILD GRAPH
# --------------------------------------------------

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

# --------------------------------------------------
# 8. RUN BOT (CLI)
# --------------------------------------------------

def run_bot():
    print("\n" + "=" * 45)
    print(" 🤖 QA LangGraph Agent (Gemini) Ready ")
    print("=" * 45)

    try:
        user_input = input("Enter your test scenario: ").strip()
        if not user_input:
            print("Error: Scenario cannot be empty.")
            return

        log_step("INPUT SCENARIO", user_input)

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )

        print("\n FINAL OUTPUT\n")
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                log_step("OBSERVATION", msg.content)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

# --------------------------------------------------
# 9. ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    run_bot()
