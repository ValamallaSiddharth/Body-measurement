"""Safe single-process defaults: the inference semaphore is process-local."""
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "5000")
workers = 1
worker_class = "gthread"
threads = 2
timeout = 150
graceful_timeout = 150
keepalive = 5
limit_request_line = 4094
limit_request_fields = 50
limit_request_field_size = 8190
max_requests = 500
max_requests_jitter = 50
accesslog = "-"
errorlog = "-"
# Avoid Gunicorn heartbeat files on persistent storage (Linux Docker).
worker_tmp_dir = "/dev/shm"
preload_app = False
