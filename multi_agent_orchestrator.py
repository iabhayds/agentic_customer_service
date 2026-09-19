from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from enum import Enum
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()  # This loads from .env
# No need to manually set OPENAI_API_KEY

# ============================================
# STEP 1: Data Structures
# ============================================

class AgentType(str, Enum):
    RETURN_AGENT = "ReturnAgent"
    WARRANTY_AGENT = "WarrantyAgent"
    REFUND_AGENT = "RefundAgent"
    POLICY_AGENT = "PolicyAgent"

@dataclass
class AgentMessage:
    """Structured message from agent to coordinator"""
    agent_type: AgentType
    response: str
    status: str  # "success", "failed", "escalate"
    tools_called: List[str]
    confidence: float
    latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self):
        return {
            "agent": self.agent_type.value,
            "response": self.response,
            "status": self.status,
            "tools_called": self.tools_called,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "error": self.error
        }

@dataclass
class Intent:
    """Detected customer intent"""
    primary: str
    secondary: Optional[str] = None
    confidence: float = 0.0
    requires_multiple_agents: bool = False

# ============================================
# STEP 2: Specialized Agents
# ============================================

class ReturnAgent:
    """Handles return requests"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_type = AgentType.RETURN_AGENT
        self.scope = ["RETURN_REQUEST"]
    
    def can_handle(self, intent: str) -> bool:
        return intent in self.scope
    
    def process(self, query: str, order_id: Optional[str] = None) -> AgentMessage:
        """Process return request"""
        start_time = datetime.now()
        
        # Simulated return logic
        tools_called = []
        
        if not order_id:
            return AgentMessage(
                agent_type=self.agent_type,
                response="I need your order ID to process your return. What's your order number?",
                status="failed",
                tools_called=tools_called,
                confidence=0.0,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error="missing_order_id"
            )
        
        # Tool 1: Lookup order
        tools_called.append("lookup_order")
        order = self._lookup_order(order_id)
        if "error" in order:
            return AgentMessage(
                agent_type=self.agent_type,
                response=f"I couldn't find order {order_id}.",
                status="failed",
                tools_called=tools_called,
                confidence=0.0,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error="order_not_found"
            )
        
        # Tool 2: Check eligibility
        tools_called.append("check_return_eligibility")
        eligibility = self._check_eligibility(order_id, order["days_since_purchase"])
        
        if eligibility["eligible"]:
            response = f"""✓ Return Approved!
Product: {order['product']} (${order['price']})
Days Since Purchase: {order['days_since_purchase']}
Refund: Full refund within 5-7 business days"""
            status = "success"
            confidence = 0.95
        else:
            response = f"""✗ Return Not Eligible
Product: {order['product']}
Days Since Purchase: {order['days_since_purchase']} (exceeds 30-day window)"""
            status = "failed"
            confidence = 0.90
        
        return AgentMessage(
            agent_type=self.agent_type,
            response=response,
            status=status,
            tools_called=tools_called,
            confidence=confidence,
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000
        )
    
    def _lookup_order(self, order_id: str) -> dict:
        """Lookup order details"""
        orders = {
            "ORD_123": {"product": "Laptop", "price": 999, "purchase_date": "2024-02-15", "days_since_purchase": 25, "status": "delivered"},
            "ORD_124": {"product": "Mouse", "price": 29, "purchase_date": "2024-01-01", "days_since_purchase": 157, "status": "delivered"},
        }
        return orders.get(order_id, {"error": "Order not found"})
    
    def _check_eligibility(self, order_id: str, days_since: int) -> dict:
        """Check return eligibility"""
        return {
            "order_id": order_id,
            "eligible": days_since <= 30,
            "max_days": 30,
            "days_elapsed": days_since
        }

class WarrantyAgent:
    """Handles warranty claims"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_type = AgentType.WARRANTY_AGENT
        self.scope = ["WARRANTY_CLAIM"]
    
    def can_handle(self, intent: str) -> bool:
        return intent in self.scope
    
    def process(self, query: str, order_id: Optional[str] = None) -> AgentMessage:
        """Process warranty claim"""
        start_time = datetime.now()
        tools_called = []
        
        response = f"""📋 Warranty Claim Initiated
Your warranty claim has been escalated to our support team.
Expected response time: 24-48 hours
Ticket: TKT_WARRANTY_{uuid.uuid4().hex[:6].upper()}"""
        
        return AgentMessage(
            agent_type=self.agent_type,
            response=response,
            status="success",
            tools_called=tools_called,
            confidence=0.88,
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

class RefundAgent:
    """Handles refund requests"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_type = AgentType.REFUND_AGENT
        self.scope = ["REFUND_REQUEST"]
    
    def can_handle(self, intent: str) -> bool:
        return intent in self.scope
    
    def process(self, query: str, order_id: Optional[str] = None) -> AgentMessage:
        """Process refund request"""
        start_time = datetime.now()
        tools_called = ["check_refund_policy"]
        
        response = f"""Refund Policy Information:
- Refund Window: 30 days from purchase
- Refund Type: Full refund
- Processing Time: 5-7 business days
- Refund Method: Original payment method

To process a refund, I'll need your order ID. What's your order number?"""
        
        return AgentMessage(
            agent_type=self.agent_type,
            response=response,
            status="success",
            tools_called=tools_called,
            confidence=0.85,
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

class PolicyAgent:
    """Handles policy questions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_type = AgentType.POLICY_AGENT
        self.scope = ["POLICY_QUESTION"]
    
    def can_handle(self, intent: str) -> bool:
        return intent in self.scope
    
    def process(self, query: str, order_id: Optional[str] = None) -> AgentMessage:
        """Answer policy questions"""
        start_time = datetime.now()
        tools_called = ["fetch_policies"]
        
        response = f"""Our Policies:
- Returns: 30-day return window, full refund
- Warranties: 1-year manufacturer warranty on all products
- Refunds: Processed within 5-7 business days
- Shipping: Free shipping on orders >$50

Is there a specific policy you'd like to know more about?"""
        
        return AgentMessage(
            agent_type=self.agent_type,
            response=response,
            status="success",
            tools_called=tools_called,
            confidence=0.92,
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

# ============================================
# STEP 3: Intent Classifier (Multi-Intent Aware)
# ============================================

def classify_intent(query: str, llm) -> Intent:
    """Classify user intent with multi-intent detection"""
    
    classification_prompt = f"""Classify the customer's intent. Output MUST be valid JSON.

Intent categories:
- RETURN_REQUEST: "I want to return" "send this back"
- REFUND_REQUEST: "I want my money back" "refund please"
- WARRANTY_CLAIM: "It broke" "defective" "doesn't work"
- POLICY_QUESTION: "What's your policy?" "How long do I have?"
- ESCALATE: "I need a manager" "This is urgent"

DETECT MULTIPLE INTENTS if present. Examples:
- "My laptop is broken AND I want to return it" → primary: WARRANTY_CLAIM, secondary: RETURN_REQUEST
- "Where's my order? Can I get a refund?" → primary: ORDER_STATUS, secondary: REFUND_REQUEST

Customer query: "{query}"

Respond ONLY with JSON (no markdown):
{{"primary_intent": "CATEGORY", "secondary_intent": null or "CATEGORY", "confidence": 0.95, "requires_multiple_agents": true/false}}"""
    
    try:
        response = llm.invoke(classification_prompt)
        content = response.content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(content)
        
        return Intent(
            primary=data.get("primary_intent", "ESCALATE"),
            secondary=data.get("secondary_intent"),
            confidence=data.get("confidence", 0.5),
            requires_multiple_agents=data.get("requires_multiple_agents", False)
        )
    except Exception as e:
        print(f"[DEBUG] Classification error: {e}")
        return Intent(
            primary="ESCALATE",
            secondary=None,
            confidence=0.3,
            requires_multiple_agents=False
        )

# ============================================
# STEP 4: Coordinator Agent
# ============================================

class CoordinatorAgent:
    """Orchestrates multi-agent system"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {
            AgentType.RETURN_AGENT: ReturnAgent(llm),
            AgentType.WARRANTY_AGENT: WarrantyAgent(llm),
            AgentType.REFUND_AGENT: RefundAgent(llm),
            AgentType.POLICY_AGENT: PolicyAgent(llm),
        }
        
        # Domain-specific thresholds (Q6: cost-aware routing)
        self.confidence_thresholds = {
            "RETURN_REQUEST": 0.70,      # Low cost of error
            "REFUND_REQUEST": 0.75,      # Medium cost
            "WARRANTY_CLAIM": 0.75,      # Medium cost
            "POLICY_QUESTION": 0.65,     # Very low cost
            "ESCALATE": 0.80             # High cost
        }
    
    def run(self, query: str, order_id: Optional[str] = None) -> Dict:
        """Main orchestration logic"""
        
        print(f"\n{'='*70}")
        print(f"[COORDINATOR] Processing: {query[:60]}...")
        print(f"{'='*70}\n")
        
        # Step 1: Detect intent(s)
        intent = classify_intent(query, self.llm)
        print(f"[INTENT DETECTION] Primary: {intent.primary} | Secondary: {intent.secondary} | Confidence: {intent.confidence:.2f}")
        
        # Step 2: Check confidence threshold (Q6: depends on cost)
        threshold = self.confidence_thresholds.get(intent.primary, 0.70)
        if intent.confidence < threshold:
            print(f"[COORDINATOR] Low confidence ({intent.confidence:.2f} < {threshold}). Asking for clarification.")
            return {
                "status": "clarification_needed",
                "response": f"I'm not entirely sure. Could you clarify: are you looking to {intent.primary.lower().replace('_', ' ')}?",
                "intent": intent.primary,
                "confidence": intent.confidence,
                "messages": [],
                "success": False
            }
        
        # Step 3: Route to agent(s) - multi-intent handling (Q1)
        messages = []
        
        # Route primary intent
        primary_agent = self._get_agent_for_intent(intent.primary)
        if primary_agent:
            print(f"[ROUTING] → {primary_agent.agent_type.value}")
            msg = primary_agent.process(query, order_id)
            messages.append(msg)
        
        # Route secondary intent if exists (multi-agent coordination)
        if intent.secondary and intent.requires_multiple_agents:
            secondary_agent = self._get_agent_for_intent(intent.secondary)
            if secondary_agent:
                print(f"[ROUTING] → {secondary_agent.agent_type.value} (secondary)")
                msg = secondary_agent.process(query, order_id)
                messages.append(msg)
        
        # Step 4: Aggregate responses
        success = all(m.status == "success" for m in messages)
        
        # Combine responses if multiple agents
        if len(messages) > 1:
            combined_response = "\n\n".join([f"[{m.agent_type.value}]\n{m.response}" for m in messages])
        else:
            combined_response = messages[0].response if messages else "No agent could handle this."
        
        # Step 5: Return orchestrated result
        return {
            "status": "success" if success else "partial",
            "response": combined_response,
            "intent": intent.primary,
            "secondary_intent": intent.secondary,
            "confidence": intent.confidence,
            "messages": [m.to_dict() for m in messages],
            "multi_agent": len(messages) > 1,
            "success": success
        }
    
    def _get_agent_for_intent(self, intent_str: str):
        """Map intent to agent"""
        mapping = {
            "RETURN_REQUEST": AgentType.RETURN_AGENT,
            "WARRANTY_CLAIM": AgentType.WARRANTY_AGENT,
            "REFUND_REQUEST": AgentType.REFUND_AGENT,
            "POLICY_QUESTION": AgentType.POLICY_AGENT,
        }
        agent_type = mapping.get(intent_str)
        if agent_type:
            return self.agents.get(agent_type)
        return None

# ============================================
# STEP 5: Run Multi-Agent System
# ============================================

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    coordinator = CoordinatorAgent(llm)
    
    test_queries = [
        ("I want to return my laptop (ORD_123)", "ORD_123"),
        ("The mouse is broken, what should I do?", None),
        ("What's your return policy?", None),
        ("My laptop screen is cracked AND I want to return it", "ORD_123"),  # Multi-intent
        ("I need a refund for my broken order", None),
    ]
    
    for query, order_id in test_queries:
        result = coordinator.run(query, order_id)
        
        print(f"\n[RESPONSE]")
        print(result["response"])
        print(f"\nMulti-Agent? {result['multi_agent']} | Success: {result['success']}")
        print(f"Agents used: {[m['agent'] for m in result['messages']]}")