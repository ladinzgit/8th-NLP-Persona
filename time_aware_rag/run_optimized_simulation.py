"""
Optimized Runner for Team 3 Simulation
---------------------------------------
This script wraps the original `simulation_model_c_async.py` to maximize resource usage (CPU/RAM).
It increases the default ThreadPoolExecutor size and the `max_concurrent` limit.
"""
import asyncio
import concurrent.futures
import sys
import os

# Add project root to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from time_aware_rag.simulation_model_c_async import run_experiment_c_rag_async

def run_optimized():
    # Configuration for maximum performance
    # Increase this based on your CPU cores and available RAM
    # Recommended: 4 * CPU_CORES or higher if IO-bound
    MAX_WORKERS = 100 
    
    print(f"🚀 Starting Optimized Runner with {MAX_WORKERS} concurrent workers...")
    
    # Customize the default thread pool executor to allow more threads
    # This is critical because we use asyncio.to_thread for DB retrieval
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    loop.set_default_executor(executor)

    try:
        # Run the experiment with higher max_concurrent
        loop.run_until_complete(
            run_experiment_c_rag_async(
                n_per_type=13,     # Default agent count
                max_concurrent=MAX_WORKERS  # Matches the thread pool size
            )
        )
    finally:
        loop.close()

if __name__ == "__main__":
    run_optimized()
