from multi_agent_orchestrator import CoordinatorAgent, Intent, classify_intent
from fault_tolerance import RetryStrategy, TimeoutManager, CircuitBreaker, ResilientAgent
from langchain_openai import ChatOpenAI
import json
from typing import List, Dict
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()  # This loads from .env
# No need to manually set OPENAI_API_KEY

# ============================================
# STEP 1: Advanced Test Dataset
# ============================================

COMPREHENSIVE_TEST_CASES = [
    # Happy path - single intent
    {
        "id": 1,
        "query": "I want to return my laptop (ORD_123)",
        "order_id": "ORD_123",
        "expected_intents": ["RETURN_REQUEST"],
        "expected_success": True,
        "category": "return_happy_path",
        "complexity": "simple"
    },
    {
        "id": 2,
        "query": "What's your return policy?",
        "order_id": None,
        "expected_intents": ["POLICY_QUESTION"],
        "expected_success": True,
        "category": "policy_happy_path",
        "complexity": "simple"
    },
    
    # Multi-intent queries (tests coordinator decomposition)
    {
        "id": 3,
        "query": "My laptop screen is cracked AND I want to return it",
        "order_id": "ORD_123",
        "expected_intents": ["WARRANTY_CLAIM", "RETURN_REQUEST"],
        "expected_success": True,
        "category": "multi_intent_complex",
        "complexity": "complex"
    },
    {
        "id": 4,
        "query": "Product is broken and I need a refund",
        "order_id": None,
        "expected_intents": ["WARRANTY_CLAIM", "REFUND_REQUEST"],
        "expected_success": True,
        "category": "multi_intent_ambiguous",
        "complexity": "complex"
    },
    
    # Edge cases
    {
        "id": 5,
        "query": "I want to return order ABC_999 (invalid order ID)",
        "order_id": "ABC_999",
        "expected_intents": ["RETURN_REQUEST"],
        "expected_success": False,
        "category": "edge_invalid_order",
        "complexity": "edge"
    },
    {
        "id": 6,
        "query": "Where's my order?",
        "order_id": None,
        "expected_intents": ["ORDER_STATUS"],
        "expected_success": False,  # No order ID provided
        "category": "edge_missing_info",
        "complexity": "edge"
    },
    {
        "id": 7,
        "query": "This is completely out of scope, tell me a joke",
        "order_id": None,
        "expected_intents": ["ESCALATE"],
        "expected_success": False,
        "category": "edge_out_of_scope",
        "complexity": "edge"
    },
    
    # Ambiguous queries (low confidence)
    {
        "id": 8,
        "query": "Product stopped working after 40 days",
        "order_id": None,
        "expected_intents": ["WARRANTY_CLAIM"],
        "expected_success": False,  # Ambiguous (warranty vs escalate)
        "category": "ambiguous_warranty",
        "complexity": "ambiguous"
    },
    
    # High-value transactions (test cost-aware routing)
    {
        "id": 9,
        "query": "I need to return my $5000 laptop (ORD_123)",
        "order_id": "ORD_123",
        "expected_intents": ["RETURN_REQUEST"],
        "expected_success": True,
        "category": "high_value_return",
        "complexity": "high_value"
    },
    
    # Customer satisfaction signals
    {
        "id": 10,
        "query": "The mouse is broken (ORD_124)",
        "order_id": "ORD_124",
        "expected_intents": ["WARRANTY_CLAIM"],
        "expected_success": True,
        "category": "warranty_with_order",
        "complexity": "simple"
    },
]

# ============================================
# STEP 2: Advanced Metrics
# ============================================

class AdvancedMetrics:
    """Sophisticated eval metrics for multi-agent systems"""
    
    @staticmethod
    def coordinator_routing_accuracy(results: List[Dict]) -> Dict:
        """How well did coordinator detect intents?"""
        correct = 0
        total = 0
        by_complexity = {}
        
        for r in results:
            total += 1
            complexity = r["test_case"]["complexity"]
            
            if complexity not in by_complexity:
                by_complexity[complexity] = {"correct": 0, "total": 0}
            
            by_complexity[complexity]["total"] += 1
            
            # Check if all detected intents are in expected
            detected = set(r["detected_intents"])
            expected = set(r["test_case"]["expected_intents"])
            
            if detected == expected:
                correct += 1
                by_complexity[complexity]["correct"] += 1
        
        overall = correct / total if total > 0 else 0
        
        by_complexity_acc = {}
        for complexity, data in by_complexity.items():
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            by_complexity_acc[complexity] = {
                "accuracy": acc,
                "count": data["total"]
            }
        
        return {
            "overall_accuracy": overall,
            "by_complexity": by_complexity_acc,
            "correct": correct,
            "total": total
        }
    
    @staticmethod
    def end_to_end_success_rate(results: List[Dict]) -> Dict:
        """Did the customer actually get solved?"""
        successful = sum(1 for r in results if r["end_to_end_success"])
        total = len(results)
        
        by_category = {}
        for r in results:
            cat = r["test_case"]["category"]
            if cat not in by_category:
                by_category[cat] = {"success": 0, "total": 0}
            
            by_category[cat]["total"] += 1
            if r["end_to_end_success"]:
                by_category[cat]["success"] += 1
        
        by_category_acc = {}
        for cat, data in by_category.items():
            acc = data["success"] / data["total"] if data["total"] > 0 else 0
            by_category_acc[cat] = {
                "success_rate": acc,
                "count": data["total"]
            }
        
        return {
            "overall_success_rate": successful / total if total > 0 else 0,
            "by_category": by_category_acc,
            "successful": successful,
            "total": total
        }
    
    @staticmethod
    def multi_agent_effectiveness(results: List[Dict]) -> Dict:
        """How well does multi-agent routing work vs single-agent?"""
        single_intent = [r for r in results if len(r["test_case"]["expected_intents"]) == 1]
        multi_intent = [r for r in results if len(r["test_case"]["expected_intents"]) > 1]
        
        single_success = sum(1 for r in single_intent if r["end_to_end_success"]) / len(single_intent) if single_intent else 0
        multi_success = sum(1 for r in multi_intent if r["end_to_end_success"]) / len(multi_intent) if multi_intent else 0
        
        return {
            "single_intent_success_rate": single_success,
            "multi_intent_success_rate": multi_success,
            "single_intent_count": len(single_intent),
            "multi_intent_count": len(multi_intent),
            "improvement": multi_success - single_success
        }
    
    @staticmethod
    def cost_analysis(results: List[Dict]) -> Dict:
        """Calculate cost of operations"""
        total_latency_ms = sum(r["latency_ms"] for r in results)
        avg_latency_ms = total_latency_ms / len(results) if results else 0
        
        # Assume: $0.001 per agent call, $0.01 per escalation, $0.1 per human intervention
        agent_calls = sum(len(r["agents_called"]) for r in results)
        escalations = sum(1 for r in results if r["escalation_needed"])
        
        cost = (agent_calls * 0.001) + (escalations * 0.01)
        
        return {
            "total_cost_usd": cost,
            "cost_per_query": cost / len(results) if results else 0,
            "average_latency_ms": avg_latency_ms,
            "total_agent_calls": agent_calls,
            "escalations": escalations
        }
    
    @staticmethod
    def failure_mode_analysis(results: List[Dict]) -> Dict:
        """Analyze failure patterns"""
        failures = [r for r in results if not r["end_to_end_success"]]
        
        failure_types = {}
        for r in failures:
            failure_type = r.get("failure_type", "unknown")
            if failure_type not in failure_types:
                failure_types[failure_type] = 0
            failure_types[failure_type] += 1
        
        return {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(results) if results else 0,
            "by_type": failure_types
        }

# ============================================
# STEP 3: Run Comprehensive Eval
# ============================================

def run_advanced_eval(coordinator: CoordinatorAgent, test_cases: List[Dict]) -> List[Dict]:
    """Run eval with full instrumentation"""
    
    results = []
    
    for test in test_cases:
        print(f"\n[EVAL {test['id']}/{len(test_cases)}] {test['query'][:50]}...")
        
        start_time = time.time()
        
        try:
            # Run coordinator
            output = coordinator.run(test["query"], test["order_id"])
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract detected intents from messages
            detected_intents = [msg["agent"].replace("Agent", "").replace("_AGENT", "") for msg in output.get("messages", [])]
            
            # Infer success
            end_to_end_success = (
                output.get("success", False) and
                set(output.get("messages", [])) != set([])
            )
            
            # Determine failure type
            failure_type = None
            if not end_to_end_success:
                if not output.get("messages"):
                    failure_type = "no_agent_response"
                elif output.get("status") == "clarification_needed":
                    failure_type = "low_confidence"
                else:
                    failure_type = "agent_failure"
            
            result = {
                "test_id": test["id"],
                "test_case": test,
                "query": test["query"],
                "detected_intents": [output.get("intent")],  # Simplified for now
                "end_to_end_success": end_to_end_success,
                "latency_ms": latency_ms,
                "agents_called": [m["agent"] for m in output.get("messages", [])],
                "escalation_needed": output.get("status") == "clarification_needed",
                "failure_type": failure_type,
                "coordinator_confidence": output.get("confidence", 0.0),
                "multi_agent_routed": output.get("multi_agent", False)
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"   ERROR: {e}")
            result = {
                "test_id": test["id"],
                "test_case": test,
                "query": test["query"],
                "detected_intents": [],
                "end_to_end_success": False,
                "latency_ms": (time.time() - start_time) * 1000,
                "agents_called": [],
                "escalation_needed": False,
                "failure_type": "system_error",
                "coordinator_confidence": 0.0,
                "multi_agent_routed": False
            }
            results.append(result)
    
    return results

# ============================================
# STEP 4: Generate Comprehensive Report
# ============================================

def print_advanced_eval_report(results: List[Dict]):
    """Print detailed eval report"""
    
    print(f"\n\n{'='*80}")
    print("MULTI-AGENT SYSTEM ADVANCED EVAL REPORT")
    print(f"{'='*80}\n")
    
    # Overall metrics
    print("📊 OVERALL SYSTEM HEALTH")
    print("-" * 80)
    
    routing_acc = AdvancedMetrics.coordinator_routing_accuracy(results)
    e2e_success = AdvancedMetrics.end_to_end_success_rate(results)
    multi_agent_eff = AdvancedMetrics.multi_agent_effectiveness(results)
    cost_analysis = AdvancedMetrics.cost_analysis(results)
    failure_analysis = AdvancedMetrics.failure_mode_analysis(results)
    
    print(f"   Coordinator Routing Accuracy: {routing_acc['overall_accuracy']:.1%}")
    print(f"   End-to-End Success Rate:      {e2e_success['overall_success_rate']:.1%}")
    print(f"   Average Latency:              {cost_analysis['average_latency_ms']:.0f}ms")
    print(f"   Failure Rate:                 {failure_analysis['failure_rate']:.1%}")
    
    # Coordinator routing by complexity
    print(f"\n🎯 COORDINATOR ROUTING ACCURACY (by complexity)")
    print("-" * 80)
    for complexity, data in routing_acc['by_complexity'].items():
        print(f"   {complexity.upper():12} {data['accuracy']:.1%} ({data['count']} tests)")
    
    # End-to-end success by category
    print(f"\n✅ END-TO-END SUCCESS (by category)")
    print("-" * 80)
    for category, data in e2e_success['by_category'].items():
        print(f"   {category:30} {data['success_rate']:.1%} ({data['count']} tests)")
    
    # Multi-agent effectiveness
    print(f"\n🤝 MULTI-AGENT EFFECTIVENESS")
    print("-" * 80)
    print(f"   Single-intent success:  {multi_agent_eff['single_intent_success_rate']:.1%} ({multi_agent_eff['single_intent_count']} tests)")
    print(f"   Multi-intent success:   {multi_agent_eff['multi_intent_success_rate']:.1%} ({multi_agent_eff['multi_intent_count']} tests)")
    print(f"   Multi-agent improvement: {multi_agent_eff['improvement']:+.1%}")
    
    # Cost analysis
    print(f"\n💰 COST ANALYSIS")
    print("-" * 80)
    print(f"   Total cost:              ${cost_analysis['total_cost_usd']:.3f}")
    print(f"   Cost per query:          ${cost_analysis['cost_per_query']:.4f}")
    print(f"   Total agent calls:       {cost_analysis['total_agent_calls']}")
    print(f"   Escalations required:    {cost_analysis['escalations']}")
    
    # Failure analysis
    print(f"\n❌ FAILURE ANALYSIS")
    print("-" * 80)
    print(f"   Total failures:          {failure_analysis['total_failures']}")
    for failure_type, count in failure_analysis['by_type'].items():
        print(f"   {failure_type:25} {count} occurrences")
    
    # Detailed results table
    print(f"\n📋 DETAILED RESULTS")
    print("-" * 80)
    print(f"{'ID':<4} {'Category':<20} {'Routing':<15} {'E2E':<5} {'Agents':<20} {'Latency':<8}")
    print("-" * 80)
    
    for r in results:
        routing_ok = "✓" if set([r["test_case"]["expected_intents"][0]]) == {r["test_case"]["expected_intents"][0]} else "✗"
        e2e_ok = "✓" if r["end_to_end_success"] else "✗"
        agents = ", ".join(r["agents_called"][:2]) if r["agents_called"] else "none"
        
        print(f"{r['test_id']:<4} {r['test_case']['category']:<20} {routing_ok:<15} {e2e_ok:<5} {agents:<20} {r['latency_ms']:>6.0f}ms")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 80)
    
    if routing_acc['overall_accuracy'] < 0.85:
        print("   ⚠️  Routing accuracy < 85%. Improve intent classification prompt.")
    
    if e2e_success['overall_success_rate'] < 0.80:
        print("   ⚠️  End-to-end success < 80%. Review failure modes below:")
        for failure_type, count in failure_analysis['by_type'].items():
            print(f"       - {failure_type}: {count} failures")
    
    if multi_agent_eff['improvement'] < 0.05:
        print("   ⚠️  Multi-agent improvement marginal. Review coordinator logic.")
    
    if cost_analysis['average_latency_ms'] > 3000:
        print(f"   ⚠️  Latency {cost_analysis['average_latency_ms']:.0f}ms > 3s threshold. Optimize tools/timeouts.")
    
    print("\n" + "="*80 + "\n")

# ============================================
# STEP 5: Run Everything
# ============================================

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    coordinator = CoordinatorAgent(llm)
    
    # Run eval
    print("Starting Advanced Multi-Agent System Eval...\n")
    results = run_advanced_eval(coordinator, COMPREHENSIVE_TEST_CASES)
    
    # Generate report
    print_advanced_eval_report(results)
    
    # Save results
    with open("eval_multi_agent_results.json", "w") as f:
        # Convert results to serializable format
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            r_copy["test_case"] = {k: v for k, v in r["test_case"].items() if not callable(v)}
            serializable_results.append(r_copy)
        
        json.dump(serializable_results, f, indent=2)
    
    print("✅ Results saved to eval_multi_agent_results.json")