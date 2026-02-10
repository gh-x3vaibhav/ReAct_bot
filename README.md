
# LangGraph ReAct Agent for Test Case Generation

## Objective
To demonstrate the **ReAct (Reason + Act)** pattern using LangGraph. The agent takes a software requirement and autonomously generates formatted test cases by reasoning through a sequence of tool calls.

## How It Works (The ReAct Pattern)
The agent follows a loop:
1.  **Thought:** Analyzes the user request.
2.  **Act:** Decides to call a specific tool (e.g., `requirement_parser`).
3.  **Observe:** Receives the output from the tool.
4.  **Repeat:** Uses the new information to decide the next step until the final answer is ready.

## Tools Implemented
1.  **`requirement_parser`**: Extracts actionable logic from natural language.
2.  **`test_case_generator`**: Creates raw test scenarios based on parsed data.
3.  **`formatter_tool`**: Converts the list of tests into a structured final report.

## Setup & Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Add API key to `.env`: `OPENAI_API_KEY=sk-...`
3.  Run the agent: `python bot.py`

## 📝 Sample Input & Output

Here is a real example of how the agent translates a raw requirement into structured test cases.

### **Input Scenario**
> "Users must be able to reset their password via email link."

### **Agent Output**
The agent analyzes the requirement, identifies the key action (*reset password*) and expected outcome (*receive email*), and generates the following report:

```text
--- QA AUTOMATION REPORT ---
TC_001: POSITIVE: Verify user can 'reset password' and see 'email link sent'.
TC_002: NEGATIVE: Verify 'reset password' with empty data does NOT show 'email link sent'.
TC_003: BOUNDARY: Verify 'reset password' with max character limit handles gracefully.
TC_004: SECURITY: Verify 'reset password' is protected against common vulnerabilities.
----------------------------=