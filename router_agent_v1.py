from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from enum import Enum
import json
from dotenv import load_dotenv
import os

load_dotenv()  # This loads from .env

# ============================================
# STEP 1: Define Intent Types
# ============================================

class Intent(str, Enum):
    RETURN_REQUEST = "RETURN_REQUEST"
    REFUND_REQUEST = "REFUND_REQUEST"
    ORDER_STATUS = "ORDER_STATUS"
    POLICY_QUESTION = "POLICY_QUESTION"
    WARRANTY_CLAIM = "WARRANTY_CLAIM"
    ESCALATE = "ESCALATE"
    UNKNOWN = "UNKNOWN"

# ============================================
# STEP 2: Define Tools (Improved from Week 1)
# ============================================

@tool
def lookup_order_details(order_id: str) -> dict:
    """Look up order details (structured return)"""
    orders = {
        "ORD_123": {
            "customer_id": "CUST_001",
            "product": "Laptop",
            "price": 999,
            "purchase_date": "2024-02-15",
            "status": "delivered",
            "days_since_purchase": 25
        },
        "ORD_124": {
            "customer_id": "CUST_002",
            "product": "Mouse",
            "price": 29,
            "purchase_date": "2024-01-01",
            "status": "delivered",
            "days_since_purchase": 157
        }
    }
    order = orders.get(order_id)
    if not order:
        return {"error": "Order not found", "order_id": order_id}
    return order

@tool
def check_return_eligibility(order_id: str, days_since_purchase: int) -> dict:
    """Check if order is eligible for return (structured)"""
    max_return_days = 30
    is_eligible = days_since_purchase <= max_return_days
    
    return {
        "order_id": order_id,
        "eligible": is_eligible,
        "max_return_days": max_return_days,
        "days_since_purchase": days_since_purchase,
        "reason": "Within return window" if is_eligible else f"Exceeded {max_return_days}-day window"
    }

@tool
def get_refund_policy() -> dict:
    """Get refund policy (structured)"""
    return {
        "refund_window_days": 30,
        "refund_type": "full",
        "processing_days": "5-7",
        "payment_method": "original_payment_method"
    }

def execute_warranty_workflow() -> dict:
    """Handle warranty claims"""
    user_message = """📋 Warranty Claim Initiated
Your warranty claim has been escalated to our support team.
Expected response time: 24-48 hours
Ticket: TKT_WARRANTY_001"""
    return {"user_message": user_message, "success": True}

@tool
def get_customer_payment_history(customer_id: str) -> dict:
    """Get customer payment history (structured)"""
    histories = {
        "CUST_001": {
            "total_purchases": 5,
            "total_spent": 3500,
            "returns_made": 0,
            "payment_status": "good_standing"
        },
        "CUST_002": {
            "total_purchases": 2,
            "total_spent": 150,
            "returns_made": 3,
            "payment_status": "flagged"
        }
    }
    return histories.get(customer_id, {"error": "Customer not found"})

@tool
def escalate_to_human() -> dict:
    """Escalate to human agent"""
    return {
        "escalated": True,
        "reason": "Complex request requiring human review",
        "ticket_id": "TKT_2024_001",
        "wait_time_minutes": 5
    }

# ============================================
# STEP 3: Intent Classifier
# ============================================

def classify_intent(query: str, llm) -> dict:
    """Classify user intent with examples"""
    classification_prompt = f"""
    Classify the customer's intent into ONE of these categories:
    
    - RETURN_REQUEST: "I want to return my laptop" "Can I send this back?"
    - REFUND_REQUEST: "Can I get my money back?" "Refund please"
    - ORDER_STATUS: "Where's my order?" "When will it arrive?"
    - POLICY_QUESTION: "What's your return policy?" "How long do I have?"
    - WARRANTY_CLAIM: "It broke after 1 day" "This is defective"
    - ESCALATE: "This is urgent" "I need to speak to someone"
    - UNKNOWN: Doesn't fit above categories
    
    Customer query: "{query}"
    
    Respond ONLY with JSON:
    {{
        "intent": "CATEGORY",
        "confidence": 0.0-1.0,
        "reasoning": "why"
    }}
    """
    
    response = llm.invoke(classification_prompt)
    
    # Parse JSON response
    try:
        result = json.loads(response.content)
        return result
    except:
        return {
            "intent": "UNKNOWN",
            "confidence": 0.5,
            "reasoning": "Could not parse classification"
        }

# ============================================
# STEP 4: Tool Chain Executor
# ============================================

def execute_return_workflow(order_id: str) -> str:
    """Execute return request workflow"""
    # Step 1: Look up order
    order = lookup_order_details.invoke({"order_id": order_id})
    if "error" in order:
        return f"Error: {order['error']}"
    
    # Step 2: Check eligibility
    eligibility = check_return_eligibility.invoke({
        "order_id": order_id,
        "days_since_purchase": order["days_since_purchase"]
    })
    
    # Step 3: Get refund policy
    policy = get_refund_policy.invoke({})
    
    # Step 4: Build response
    if eligibility["eligible"]:
        return f"""
✓ Return Approved!
Product: {order['product']} (${order['price']})
Purchase Date: {order['purchase_date']}
Days Since Purchase: {order['days_since_purchase']} (within {policy['refund_window_days']}-day window)
Refund Type: {policy['refund_type']} refund
Processing Time: {policy['processing_days']} business days
        """
    else:
        return f"""
✗ Return Not Eligible
Product: {order['product']}
Days Since Purchase: {order['days_since_purchase']} (exceeds {policy['refund_window_days']}-day window)
Reason: {eligibility['reason']}
        """

def execute_order_status_workflow(order_id: str) -> str:
    """Execute order status workflow"""
    order = lookup_order_details.invoke({"order_id": order_id})
    if "error" in order:
        return f"Error: {order['error']}"
    
    return f"""
Order Status for {order_id}:
Product: {order['product']} (${order['price']})
Status: {order['status'].upper()}
Purchased: {order['purchase_date']}
    """

def execute_policy_workflow() -> str:
    """Execute policy question workflow"""
    policy = get_refund_policy.invoke({})
    return f"""
Return & Refund Policy:
- Return Window: {policy['refund_window_days']} days from purchase
- Refund Type: {policy['refund_type']} refund
- Processing Time: {policy['processing_days']} business days
- Refund Method: {policy['payment_method']}
    """

# ============================================
# STEP 5: Router Agent
# ============================================

class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, query: str) -> dict:
        """Main router logic"""
         # Initialize result (so it's always defined)
        result = {"user_message": "Something went wrong. Please try again.", "success": False}
        
        # Step 1: Classify intent
        classification = classify_intent(query, self.llm)
        intent = Intent(classification.get("intent", "UNKNOWN"))
        confidence = classification.get("confidence", 0.0)
        
        print(f"\n[ROUTER] Intent: {intent.value} (confidence: {confidence})")
        
        # Step 2: Route based on intent
        if intent == Intent.RETURN_REQUEST:
            # Extract order ID from query (simple heuristic)
            order_id = self._extract_order_id(query)
            if not order_id:
                response = "I need an order ID to process your return. What's your order number?"
            else:
                response = execute_return_workflow(order_id)
        
        elif intent == Intent.ORDER_STATUS:
            order_id = self._extract_order_id(query)
            if not order_id:
                response = "I need an order ID to check status. What's your order number?"
            else:
                response = execute_order_status_workflow(order_id)
        
        elif intent == Intent.POLICY_QUESTION:
            response = execute_policy_workflow()

        elif intent == Intent.WARRANTY_CLAIM:
            response = execute_warranty_workflow()
        
        elif intent == Intent.ESCALATE:
            result = escalate_to_human.invoke({})
            response = f"Escalating to human agent. Ticket: {result['ticket_id']}. Wait time: {result['wait_time_minutes']} min"
        
        else:
            response = "I'm not sure how to help with that. Can you clarify? (Returns, refunds, order status, or policies)"
        
        # Step 3: Return result
        return {
            "intent": intent.value,
            "confidence": confidence,
            "response": response,
            #"user_message": result.get("user_message", ""),
            "success": result.get("success", False)
        }
    
    def _extract_order_id(self, query: str) -> str:
        """Simple order ID extractor"""
        import re
        match = re.search(r'ORD_\d+', query)
        return match.group(0) if match else None

# ============================================
# STEP 6: Run Router Agent
# ============================================

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    router = RouterAgent(llm)
    
    test_queries = [
        "I want to return my laptop (ORD_123)",
        "What's the status of order ORD_124?",
        "What's your return policy?",
        "I need a refund for ORD_123",
        "The mouse broke after 1 day, it's still under warranty",
        "My printer stopped working after 40 days, what should i do?",
        "How do i find out warranty details of laptop"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        result = router.run(query)
        
        
        print(f"\n[RESPONSE]\n{result['response']}")

       