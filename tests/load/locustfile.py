"""Load testing scripts using Locust.

Features:
- API endpoint load testing
- Configurable user patterns
- Response time tracking
- Error rate monitoring
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

try:
    from locust import HttpUser, between, task, events
    from locust.runners import MasterRunner

    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

if LOCUST_AVAILABLE:

    class CADMLPlatformUser(HttpUser):
        """Simulated user for CAD ML Platform load testing."""

        # Wait 1-3 seconds between tasks
        wait_time = between(1, 3)

        # Configuration
        api_key = os.environ.get("API_KEY", "load-test-api-key")

        def on_start(self):
            """Setup before starting tasks."""
            self.client.headers.update({
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            })

        @task(10)
        def health_check(self):
            """High frequency health check."""
            with self.client.get("/health", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Health check failed: {response.status_code}")

        @task(5)
        def api_health(self):
            """API v1 health endpoint."""
            with self.client.get("/api/v1/health", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"API health failed: {response.status_code}")

        @task(3)
        def metrics_endpoint(self):
            """Prometheus metrics endpoint."""
            with self.client.get("/metrics", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Metrics failed: {response.status_code}")

        @task(2)
        def material_classify(self):
            """Material classification endpoint."""
            payload = {
                "features": {
                    "thickness": random.uniform(1.0, 10.0),
                    "surface_area": random.uniform(100, 5000),
                    "has_holes": random.choice([True, False]),
                    "hole_count": random.randint(0, 20),
                },
                "text_hints": random.sample(
                    ["steel", "aluminum", "AISI 304", "6061-T6", "brass"],
                    k=random.randint(1, 3),
                ),
            }

            with self.client.post(
                "/api/v2/materials/classify",
                json=payload,
                catch_response=True,
            ) as response:
                if response.status_code in [200, 201]:
                    response.success()
                elif response.status_code == 404:
                    response.success()  # Endpoint might not exist
                else:
                    response.failure(f"Material classify failed: {response.status_code}")

        @task(1)
        def batch_status_check(self):
            """Check batch job status (simulated)."""
            job_id = f"test-job-{random.randint(1, 1000)}"

            with self.client.get(
                f"/api/v2/batch/status/{job_id}",
                catch_response=True,
            ) as response:
                # 404 is expected for non-existent jobs
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Batch status failed: {response.status_code}")

    class HighLoadUser(HttpUser):
        """High frequency user for stress testing."""

        wait_time = between(0.1, 0.5)

        api_key = os.environ.get("API_KEY", "load-test-api-key")

        def on_start(self):
            self.client.headers.update({
                "X-API-Key": self.api_key,
            })

        @task
        def rapid_health_check(self):
            """Rapid health checks for stress testing."""
            self.client.get("/health")

    class APIConsumerUser(HttpUser):
        """Simulated API consumer with realistic patterns."""

        wait_time = between(2, 5)

        api_key = os.environ.get("API_KEY", "load-test-api-key")

        def on_start(self):
            self.client.headers.update({
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            })

        @task(5)
        def realistic_workflow(self):
            """Simulate realistic API usage workflow."""
            # Step 1: Health check
            self.client.get("/health")

            # Step 2: Get some data (simulated)
            self.client.get("/api/v1/health")

        @task(2)
        def burst_requests(self):
            """Simulate burst of requests."""
            for _ in range(5):
                self.client.get("/health")

    # Event hooks for reporting
    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        """Log test start."""
        print("=" * 60)
        print("CAD ML Platform Load Test Started")
        print("=" * 60)

    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """Log test completion with summary."""
        print("=" * 60)
        print("CAD ML Platform Load Test Completed")
        print("=" * 60)

        if environment.stats.total.num_requests > 0:
            print(f"Total Requests: {environment.stats.total.num_requests}")
            print(f"Total Failures: {environment.stats.total.num_failures}")
            print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
            print(f"Requests/sec: {environment.stats.total.current_rps:.2f}")

else:
    # Stub classes when Locust is not available
    class CADMLPlatformUser:
        pass

    class HighLoadUser:
        pass

    class APIConsumerUser:
        pass


# K6 script template for alternative load testing
K6_SCRIPT = """
// k6 load test script for CAD ML Platform
// Run with: k6 run tests/load/locustfile.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const healthCheckDuration = new Trend('health_check_duration');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up
    { duration: '3m', target: 10 },   // Steady state
    { duration: '1m', target: 50 },   // Spike
    { duration: '2m', target: 50 },   // High load
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests under 500ms
    errors: ['rate<0.1'],               // Error rate under 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json',
};

export default function () {
  // Health check
  const healthStart = Date.now();
  const healthRes = http.get(`${BASE_URL}/health`, { headers });
  healthCheckDuration.add(Date.now() - healthStart);

  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 200ms': (r) => r.timings.duration < 200,
  });

  errorRate.add(healthRes.status !== 200);

  sleep(1);

  // API health
  const apiHealthRes = http.get(`${BASE_URL}/api/v1/health`, { headers });
  check(apiHealthRes, {
    'api health status is 200': (r) => r.status === 200,
  });

  errorRate.add(apiHealthRes.status !== 200);

  sleep(1);

  // Material classification (if endpoint exists)
  const materialPayload = JSON.stringify({
    features: {
      thickness: Math.random() * 10,
      surface_area: Math.random() * 5000,
      has_holes: Math.random() > 0.5,
      hole_count: Math.floor(Math.random() * 20),
    },
    text_hints: ['steel', 'aluminum'].slice(0, Math.floor(Math.random() * 2) + 1),
  });

  const materialRes = http.post(
    `${BASE_URL}/api/v2/materials/classify`,
    materialPayload,
    { headers }
  );

  check(materialRes, {
    'material classify status is 2xx or 404': (r) =>
      r.status >= 200 && r.status < 300 || r.status === 404,
  });

  sleep(2);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'reports/load_test_results.json': JSON.stringify(data),
  };
}
"""


def generate_k6_script():
    """Generate K6 script file."""
    script_path = os.path.join(os.path.dirname(__file__), "k6_load_test.js")
    with open(script_path, "w") as f:
        f.write(K6_SCRIPT)
    print(f"K6 script generated at: {script_path}")
    return script_path


if __name__ == "__main__":
    if LOCUST_AVAILABLE:
        print("Locust is available. Run with:")
        print("  locust -f tests/load/locustfile.py --host http://localhost:8000")
    else:
        print("Locust not installed. Install with: pip install locust")

    print("\nGenerating K6 script...")
    generate_k6_script()
    print("Run K6 with: k6 run tests/load/k6_load_test.js")
