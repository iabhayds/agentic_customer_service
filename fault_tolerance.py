import time
from typing import Callable, Any, Optional
from datetime import datetime
import random
from dotenv import load_dotenv

load_dotenv()  # This loads from .env
# No need to manually set OPENAI_API_KEY

# ============================================
# STEP 1: Retry Logic with Exponential Backoff
# ============================================

class RetryStrategy:
    """Implements retry logic for tool failures"""
    
    def __init__(self, max_retries: int = 3, initial_delay_ms: int = 100):
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
    
    def execute_with_retry(self, 
                          func: Callable, 
                          *args, 
                          **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Execute function with exponential backoff retry.
        
        Returns: (success, result, error)
        """
        
        error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    print(f"   ✓ Succeeded on attempt {attempt + 1}")
                return (True, result, None)
            
            except Exception as e:
                error = str(e)
                
                if attempt < self.max_retries:
                    # Calculate backoff: 100ms, 200ms, 400ms
                    delay_ms = self.initial_delay_ms * (2 ** attempt)
                    delay_s = delay_ms / 1000
                    
                    print(f"   ✗ Attempt {attempt + 1} failed: {error}")
                    print(f"   ⏳ Retrying in {delay_s:.2f}s...")
                    time.sleep(delay_s)
                else:
                    print(f"   ✗ All {self.max_retries + 1} attempts failed.")
                    return (False, None, error)
        
        return (False, None, error)

# ============================================
# STEP 2: Timeout Manager
# ============================================

import threading

class TimeoutManager:
    """Manages agent execution timeouts"""
    
    @staticmethod
    def execute_with_timeout(func: Callable, 
                            timeout_seconds: float,
                            *args, 
                            **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Execute function with timeout.
        
        Returns: (success, result, error)
        """
        
        result = {"value": None, "error": None, "completed": False}
        
        def target():
            try:
                result["value"] = func(*args, **kwargs)
                result["completed"] = True
            except Exception as e:
                result["error"] = str(e)
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if result["completed"]:
            if result["error"]:
                return (False, None, result["error"])
            return (True, result["value"], None)
        else:
            return (False, None, f"Timeout after {timeout_seconds}s")

# ============================================
# STEP 3: Circuit Breaker (Fault Isolation)
# ============================================

class CircuitBreaker:
    """Prevents cascading failures - isolates failing agents"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Call function with circuit breaker protection.
        
        States:
        - CLOSED: Normal operation
        - OPEN: Agent failing, reject calls
        - HALF_OPEN: Testing if agent recovered
        """
        
        # If OPEN, check if recovery timeout passed
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout_s:
                print(f"   [CIRCUIT BREAKER] Attempting recovery (HALF_OPEN)")
                self.state = "HALF_OPEN"
            else:
                return (False, None, f"Circuit breaker OPEN (agent failing). Try again in {self.recovery_timeout_s}s")
        
        # Try to execute
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if HALF_OPEN
            if self.state == "HALF_OPEN":
                print(f"   [CIRCUIT BREAKER] Recovery successful! (CLOSED)")
                self.state = "CLOSED"
                self.failure_count = 0
            
            return (True, result, None)
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print(f"   [CIRCUIT BREAKER] Too many failures ({self.failure_count}). OPEN.")
            
            return (False, None, str(e))

# ============================================
# STEP 4: Graceful Degradation
# ============================================

class GracefulDegradation:
    """Handle failures by returning cached/partial results"""
    
    def __init__(self):
        self.cache = {}
    
    def execute_with_fallback(self,
                             primary_func: Callable,
                             fallback_func: Callable,
                             cache_key: str,
                             *args,
                             **kwargs) -> tuple[bool, Any, Optional[str], str]:
        """
        Execute primary function.
        On failure, use cached result or fallback.
        
        Returns: (success, result, error, source)
        source = "primary", "cache", or "fallback"
        """
        
        # Try primary
        try:
            result = primary_func(*args, **kwargs)
            self.cache[cache_key] = result
            return (True, result, None, "primary")
        except Exception as e:
            print(f"   Primary failed: {e}")
        
        # Try cache
        if cache_key in self.cache:
            print(f"   Using cached result")
            return (True, self.cache[cache_key], None, "cache")
        
        # Try fallback
        try:
            result = fallback_func(*args, **kwargs)
            print(f"   Using fallback result")
            return (True, result, None, "fallback")
        except Exception as e2:
            return (False, None, f"Primary failed: {e}, Fallback failed: {e2}", "failed")

# ============================================
# STEP 5: Production-Ready Agent Wrapper
# ============================================

class ResilientAgent:
    """Wraps an agent with fault tolerance"""
    
    def __init__(self, agent, agent_name: str, timeout_s: float = 5.0):
        self.agent = agent
        self.agent_name = agent_name
        self.timeout_s = timeout_s
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_s=60)
        self.retry_strategy = RetryStrategy(max_retries=3, initial_delay_ms=100)
        self.degradation = GracefulDegradation()
    
    def process_with_resilience(self, query: str, order_id: Optional[str] = None):
        """Process with full fault tolerance"""
        
        print(f"\n[{self.agent_name}] Processing with resilience...")
        
        # Step 1: Circuit breaker
        def protected_process():
            # Step 2: Timeout
            success, result, error = TimeoutManager.execute_with_timeout(
                self.agent.process,
                timeout_seconds=self.timeout_s,
                query=query,
                order_id=order_id
            )
            
            if not success:
                raise Exception(error)
            return result
        
        success, agent_msg, error = self.circuit_breaker.call(protected_process)
        
        if not success:
            print(f"   [Circuit Breaker] Failed: {error}")
            return {
                "status": "failed",
                "response": f"The {self.agent_name} is temporarily unavailable. Please try again in a moment.",
                "error": error,
                "resilience_applied": "circuit_breaker"
            }
        
        return {
            "status": "success" if agent_msg.status == "success" else agent_msg.status,
            "response": agent_msg.response,
            "tools_called": agent_msg.tools_called,
            "confidence": agent_msg.confidence,
            "latency_ms": agent_msg.latency_ms,
            "resilience_applied": None
        }

# ============================================
# STEP 6: Demo - Fault Tolerance in Action
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("FAULT TOLERANCE PATTERNS DEMO")
    print("="*70)
    
    # Demo 1: Retry Logic
    print("\n[DEMO 1] Retry with Exponential Backoff")
    print("-" * 70)
    
    def flaky_tool():
        """Tool that fails 2/3 times"""
        if random.random() < 0.66:
            raise Exception("Network timeout")
        return {"result": "success"}
    
    retry = RetryStrategy(max_retries=3, initial_delay_ms=100)
    success, result, error = retry.execute_with_retry(flaky_tool)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}\n")
    
    # Demo 2: Timeout
    print("[DEMO 2] Timeout Management")
    print("-" * 70)
    
    def slow_tool():
        """Tool that takes 3 seconds"""
        time.sleep(3)
        return {"result": "done"}
    
    success, result, error = TimeoutManager.execute_with_timeout(
        slow_tool,
        timeout_seconds=1
    )
    print(f"Result: {'SUCCESS' if success else 'TIMEOUT'} - {error}\n")
    
    # Demo 3: Circuit Breaker
    print("[DEMO 3] Circuit Breaker (Fault Isolation)")
    print("-" * 70)
    
    def broken_agent():
        """Agent that always fails"""
        raise Exception("Agent crashed")
    
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_s=2)
    
    for i in range(5):
        print(f"Call {i+1}:")
        success, result, error = breaker.call(broken_agent)
        if not success:
            print(f"  {error}")
        time.sleep(0.5)
    
    print(f"\nCircuit State: {breaker.state}")
    print(f"Failure Count: {breaker.failure_count}\n")
    
    # Demo 4: Graceful Degradation
    print("[DEMO 4] Graceful Degradation (Cache + Fallback)")
    print("-" * 70)
    
    def primary():
        raise Exception("Primary failed")
    
    def fallback():
        return {"source": "fallback", "data": "cached_data"}
    
    degradation = GracefulDegradation()
    success, result, error, source = degradation.execute_with_fallback(
        primary, fallback, "key1"
    )
    print(f"Success: {success} | Source: {source} | Result: {result}\n")