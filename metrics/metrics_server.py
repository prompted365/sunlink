from prometheus_client import Gauge, start_http_server
import time
import random

# Define a custom metric (e.g., task_queue_depth)
task_queue_depth = Gauge('task_queue_depth', 'Depth of the task queue')

# Simulate queue depth updates
def simulate_queue_depth():
    while True:
        # Simulate a random queue depth
        depth = random.randint(0, 20)
        task_queue_depth.set(depth)  # Update the metric
        time.sleep(5)  # Sleep for 5 seconds

if __name__ == "__main__":
    # Start the Prometheus metrics server on port 8000
    start_http_server(8000)
    print("Prometheus metrics server running on http://0.0.0.0:8000/metrics")
    simulate_queue_depth()