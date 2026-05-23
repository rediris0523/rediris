import time
import logging
import sys
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def _log_request(message: str, level: str = "INFO"):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    try:
        from rediris.common.utils.logging import get_module_prefix
        prefix = get_module_prefix()
        if prefix:
            formatted = f"{timestamp} - [{prefix}] - HTTP - {level} - {message}"
        else:
            formatted = f"{timestamp} - HTTP - {level} - {message}"
    except:
        formatted = f"{timestamp} - HTTP - {level} - {message}"

    print(formatted, flush=True)


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, exclude_paths: list = None):

        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""

        start_time = time.time()
        _log_request(f">>> {method} {path}{('?' + query_params) if query_params else ''} - Client: {client_ip}")

        try:
            response = await call_next(request)

            process_time = time.time() - start_time
            process_time_ms = round(process_time * 1000, 2)

            status_code = response.status_code
            log_level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"

            log_message = f"<<< {method} {path} - Status: {status_code} - Time: {process_time_ms}ms"
            _log_request(log_message, log_level)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            process_time_ms = round(process_time * 1000, 2)
            _log_request(f"<<< {method} {path} - Error: {str(e)} - Time: {process_time_ms}ms", "ERROR")
            raise


def add_request_logging(app, exclude_paths: list = None):
    app.add_middleware(RequestLoggingMiddleware, exclude_paths=exclude_paths)
