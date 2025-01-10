import os
import ssl
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

BROKER_URL = os.getenv("UPSTASH_BROKER_URL")
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
    include=["tasks.worker_tasks"],
    broker_connection_retry_on_startup=True
)