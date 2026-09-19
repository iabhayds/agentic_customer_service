from langchain_openai import ChatOpenAI
from router_agent_v1 import RouterAgent, Intent
import json
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================
# STEP 1: Define Test Dataset
# ============================================

TEST_CASES = [
    {
        "query": "I want to return my laptop (ORD_123)",
        "expected_intent": "RETURN_REQUEST",
        "expected_success": True,
        "category": "return_happy_path"
    },
    {
        "query": "What's the status of order ORD_124?",
        "expected_intent": "ORDER_STATUS",
        "expected_success": True,
        "category": "order_status_happy_path"
    },
    {
        "query": "What's your return policy?",
        "expected_intent": "POLICY_QUESTION",
        "expected_success": True,
        "category": "policy_question_happy_path"
    },
    {
        "query": "The mouse broke after 1 day, it's still under warranty",
        "expected_intent": "WARRANTY_CLAIM",
        "expected_success": True,
        "category": "warranty_claim_happy_path"
    },
    {
        "query": "My printer stopped working after 40 days, what should I do?",
        "expected_intent": "ESCALATE",  # Should escalate due to ambiguity
        "expected_success": False,
        "category": "ambiguous_edge_case"
    },
    {
        "query": "I need a refund for ORD_123",
        "expected_intent": "REFUND_REQUEST",
        "expected_success": True,
        "category": "refund_happy_path"
    },
    {
        "query": "This product is defective",
        "expected_intent": "WARRANTY_CLAIM",
        "expected_success": False,  # No order ID, should ask for clarification
        "category": "incomplete_info"
    },
    {
        "query": "I need to speak to someone immediately",
        "expected_intent": "ESCALATE",
        "expected_success": True,
        "category": "escalation_urgent"
    },
    {
        "query": "Tell me a joke about customer service",
        "expected_intent": "UNKNOWN",
        "expected_success": False,
        "category": "out_of_scope"
    },
    {
        "query": "I want to return order ABC_999",
        "expected_intent": "RETURN_REQUEST",
        "expected_success": False,  # Invalid order ID format
        "category": "invalid_order_format"
    }
]

# ============================================
# STEP 2: Eval Metrics
# ============================================

class EvalMetrics:
    """Calculate eval results"""
    
    @staticmethod
    def intent_accuracy(results: List[Dict]) -> float:
        """What % of intents were classified correctly?"""
        correct = sum(1 for r in results if r["intent_correct"])
        return correct / len(results) if results else 0.0
    
    @staticmethod
    def confidence_calibration(results: List[Dict]) -> Dict:
        """Is model confidence aligned with accuracy?
        
        E.g., if it says 0.9 confidence on 10 queries, 
        ~9 should be correct.
        """
        buckets = {
            "high": {"pred": 0, "correct": 0},      # 0.8-1.0
            "medium": {"pred": 0, "correct": 0},    # 0.5-0.8
            "low": {"pred": 0, "correct": 0}        # <0.5
        }
        
        for r in results:
            conf = r["confidence"]
            if conf >= 0.8:
                bucket = "high"
            elif conf >= 0.5:
                bucket = "medium"
            else:
                bucket = "low"
            
            buckets[bucket]["pred"] += 1
            if r["intent_correct"]:
                buckets[bucket]["correct"] += 1
        
        # Calculate accuracy per confidence bucket
        calibration = {}
        for bucket, data in buckets.items():
            if data["pred"] > 0:
                calibration[bucket] = {
                    "accuracy": data["correct"] / data["pred"],
                    "count": data["pred"]
                }
        
        return calibration
    
    @staticmethod
    def response_quality(results: List[Dict]) -> float:
        """What % of responses were helpful?"""
        helpful = sum(1 for r in results if r["response_helpful"])
        return helpful / len(results) if results else 0.0
    
    @staticmethod
    def error_analysis(results: List[Dict]) -> Dict:
        """Break down where the router fails"""
        errors = {}
        for r in results:
            if not r["intent_correct"]:
                key = f"{r['expected_intent']} → {r['actual_intent']}"
                errors[key] = errors.get(key, 0) + 1
        return errors

# ============================================
# STEP 3: Run Eval
# ============================================

def run_eval(router: RouterAgent, test_cases: List[Dict]) -> Dict:
    """Run evals on test cases"""
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[EVAL {i}/{len(test_cases)}] {test['query'][:50]}...")
        
        try:
            # Run router
            output = router.run(test["query"])
            
            # Score this result
            intent_correct = output.get("intent") == test["expected_intent"]
            actual_success = output.get("success", False)  # Safely get success
            response_helpful = actual_success == test["expected_success"]
            
            result = {
                "test_id": i,
                "query": test["query"],
                "category": test["category"],
                "expected_intent": test["expected_intent"],
                "actual_intent": output.get("intent", "ERROR"),
                "intent_correct": intent_correct,
                "confidence": output.get("confidence", 0.0),
                "expected_success": test["expected_success"],
                "actual_success": actual_success,
                "response_helpful": response_helpful,
                "reasoning": output.get("reasoning", "")
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"   ⚠️  Error running test: {e}")
            result = {
                "test_id": i,
                "query": test["query"],
                "category": test["category"],
                "expected_intent": test["expected_intent"],
                "actual_intent": "ERROR",
                "intent_correct": False,
                "confidence": 0.0,
                "expected_success": test["expected_success"],
                "actual_success": False,
                "response_helpful": False,
                "reasoning": f"Error: {str(e)}"
            }
            results.append(result)
    
    return results

# ============================================
# STEP 4: Report Results
# ============================================

def print_eval_report(results: List[Dict]):
    """Print human-readable eval report"""
    
    print(f"\n\n{'='*70}")
    print("ROUTER AGENT EVAL REPORT")
    print(f"{'='*70}\n")
    
    # Overall metrics
    intent_acc = EvalMetrics.intent_accuracy(results)
    response_qual = EvalMetrics.response_quality(results)
    
    print(f"📊 OVERALL METRICS")
    print(f"   Intent Accuracy:    {intent_acc:.1%} ({sum(1 for r in results if r['intent_correct'])}/{len(results)})")
    print(f"   Response Quality:   {response_qual:.1%}")
    
    # Confidence calibration
    print(f"\n📈 CONFIDENCE CALIBRATION")
    calibration = EvalMetrics.confidence_calibration(results)
    for bucket, data in calibration.items():
        print(f"   {bucket.upper():6} confidence: {data['accuracy']:.1%} accurate ({data['count']} queries)")
    
    # Error analysis
    errors = EvalMetrics.error_analysis(results)
    if errors:
        print(f"\n❌ ERROR ANALYSIS")
        for error, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error}: {count} misclassifications")
    else:
        print(f"\n✅ No misclassifications!")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS")
    print(f"{'ID':<3} {'Category':<25} {'Expected':<15} {'Actual':<15} {'Correct':<7} {'Conf':<5}")
    print(f"{'-'*70}")
    
    for r in results:
        status = "✓" if r["intent_correct"] else "✗"
        print(f"{r['test_id']:<3} {r['category']:<25} {r['expected_intent']:<15} {r['actual_intent']:<15} {status:<7} {r['confidence']:.2f}")
    
    # Failures breakdown
    failures = [r for r in results if not r["intent_correct"]]
    if failures:
        print(f"\n🔍 FAILURES ({len(failures)} total)")
        for r in failures:
            print(f"\n   Test {r['test_id']}: {r['query']}")
            print(f"   Expected: {r['expected_intent']}")
            print(f"   Got:      {r['actual_intent']} (confidence: {r['confidence']:.2f})")
            print(f"   Reasoning: {r['reasoning']}")

# ============================================
# STEP 5: Run Full Eval
# ============================================

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    router = RouterAgent(llm)
    
    # Run eval
    results = run_eval(router, TEST_CASES)
    
    # Print report
    print_eval_report(results)
    
    # Save results to JSON for later analysis
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Eval results saved to eval_results.json")