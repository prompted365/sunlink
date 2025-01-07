# tasks/celery_app.py

import os
import ssl
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Upstash Redis broker URL. Example format:
# rediss://:<PASSWORD>@<UPSTASH_REDIS_ENDPOINT>:<PORT>
#
# In production, store this securely (e.g., in an .env file or secret manager)
BROKER_URL = os.getenv(
    "UPSTASH_BROKER_URL",
    "rediss://:YOUR_UPSTASH_REDIS_PASSWORD@YOUR_UPSTASH_REDIS_HOST:30398"
)

# We can use the same URL for the backend, or omit "backend="
# if we do not need result tracking.
BACKEND_URL = BROKER_URL

celery_app = Celery("Sunlink Celery", broker=BROKER_URL, backend=BACKEND_URL)

celery_app.conf.update(
    task_track_started=True,
    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    # Add this line to explicitly include your module
    include=["tasks.worker_tasks"],
    broker_connection_retry_on_startup=True
)

