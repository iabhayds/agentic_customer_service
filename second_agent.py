from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Annotated
from typing_extensions import TypedDict
import json
from dotenv import load_dotenv
import os

load_dotenv()  # This loads from .env
# No need to manually set OPENAI_API_KEY

# ============================================
# STEP 1: Define Your Tools
# ============================================

@tool
def lookup_payment_history(customer_id: str) -> str:
    """Look up customer's payment history"""
    payment_data = {
        "CUST_001": "Payments: $50 (Jan), $75 (Feb), $100 (Mar). Status: Good",
        "CUST_002": "Payments: $25 (Jan). Status: Pending"
    }
    return payment_data.get(customer_id, "Customer not found")

@tool
def check_delivery_status(order_id: str) -> str:
    """Check where an order is in delivery"""
    delivery_data = {
        "ORD_123": "Order in transit. Expected delivery: 2 days",
        "ORD_124": "Order delivered on Mar 15",
        "ORD_125": "Order processing. Expected shipment: 1 day"
    }
    return delivery_data.get(order_id, "Order not found")

@tool
def check_hr_policies(policy_type: str) -> str:
    """Check HR policies (returns, refunds, warranties)"""
    policies = {
        "returns": "30-day return window from purchase date. Full refund.",
        "refunds": "Refund processed within 5-7 business days to original payment method",
        "warranty": "1-year manufacturer warranty on all products"
    }
    return policies.get(policy_type, "Policy not found")

@tool
def check_flags(customer_id: str) -> str:
    """Check for fraud/risk flags on customer"""
    flags = {
        "CUST_001": "No flags. Good standing.",
        "CUST_002": "Flag: Multiple refund requests in 30 days. Review recommended."
    }
    return flags.get(customer_id, "No data found")

@tool
def product_catalog(product: str) -> str:
    """check for product availability"""
    products = {
        "TV": "All models available in all sized",
        "Fridge": "Only Double Door fridges available. All other models coming by end of week"
    }
    return products.get(product, "No data found")

# Map tool names to actual functions
TOOLS = {
    "lookup_payment_history": lookup_payment_history,
    "check_delivery_status": check_delivery_status,
    "check_hr_policies": check_hr_policies,
    "check_flags": check_flags,
    "product_catalog": product_catalog
}

# ============================================
# STEP 2: Define Agent State
# ============================================

class AgentState(TypedDict):
    messages: list
    iterations: int

# ============================================
# STEP 3: Initialize LLM
# ============================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([
    lookup_payment_history,
    check_delivery_status,
    check_hr_policies,
    check_flags,
    product_catalog
])

# ============================================
# STEP 4: Define Agent Nodes
# ============================================

def agent_node(state: AgentState):
    """LLM decides what to do (think)"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response], "iterations": state["iterations"] + 1}

def tool_node(state: AgentState):
    """Execute tools (act)"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract tool calls from LLM response
    tool_calls = last_message.tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Execute tool
        if tool_name in TOOLS:
            result = TOOLS[tool_name].invoke(tool_args)
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": str(result)
            })
    
    # Add results back to messages
    from langchain_core.messages import ToolMessage
    tool_messages = [ToolMessage(content=r["content"], tool_call_id=r["tool_use_id"]) for r in results]
    
    return {"messages": messages + tool_messages, "iterations": state["iterations"]}

def should_continue(state: AgentState):
    """Decide: keep looping or stop?"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Stop if no tool calls (agent is done)
    if not hasattr(last_message, 'tool_calls') or len(last_message.tool_calls) == 0:
        return "end"
    
    # Stop after 10 iterations to prevent infinite loops
    if state["iterations"] > 10:
        return "end"
    
    # Keep going
    return "continue"

# ============================================
# STEP 5: Build the Graph
# ============================================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# Compile
agent = workflow.compile()

# ============================================
# STEP 6: Run the Agent
# ============================================

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    test_queries = [
        "What's the delivery status of order ORD_113?",
        "Customer CUST_001 wants to return an item. Can they? What's the policy?",
        "Check if customer CUST_002 has any flags and their payment history",
        "Check if Fridge is available from catalog"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "iterations": 0
        }
        
        result = agent.invoke(initial_state)
        
        # Get final response
        final_message = result["messages"][-1]
        print(f"\nAGENT RESPONSE:\n{final_message.content}")