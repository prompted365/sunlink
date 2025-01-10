from prometheus_client import Gauge, start_http_server
import time
import redis
import os

# Load the Redis URL from environment variables
REDIS_URL = os.getenv("UPSTASH_BROKER_URL")

# Define the custom metric
task_queue_depth = Gauge('task_queue_depth', 'Depth of the task queue')

# Connect to Upstash Redis
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

# Update task queue depth dynamically
def update_task_queue_depth():
    while True:
        try:
            # Example: Check queue length in Redis
            depth = redis_client.llen('task_queue')  # Replace 'task_queue' with your queue key
            task_queue_depth.set(depth)  # Update the metric
            print(f"Updated task_queue_depth to {depth}")
        except Exception as e:
            print(f"Error updating task_queue_depth: {e}")
            task_queue_depth.set(0)  # Set to 0 if an error occurs
        time.sleep(5)  # Adjust interval as needed

if __name__ == "__main__":
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    print("Prometheus metrics server running on http://0.0.0.0:8000/metrics")
    update_task_queue_depth()