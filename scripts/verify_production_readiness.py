import time
import urllib.request
import concurrent.futures
import statistics

URL = "http://localhost:8000/health"
CONCURRENCY = 10
REQUESTS = 100

def make_request():
    start = time.time()
    try:
        with urllib.request.urlopen(URL) as response:
            response.read()
            code = response.getcode()
    except Exception as e:
        code = 500
    duration = (time.time() - start) * 1000
    return code, duration

def run_load_test():
    print(f"Starting load test: {REQUESTS} requests, {CONCURRENCY} concurrency")
    latencies = []
    errors = 0
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(make_request) for _ in range(REQUESTS)]
        for future in concurrent.futures.as_completed(futures):
            code, duration = future.result()
            if code != 200:
                errors += 1
            latencies.append(duration)
    
    total_time = time.time() - start_time
    qps = REQUESTS / total_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"QPS: {qps:.2f}")
    print(f"Errors: {errors}")
    print(f"P50 Latency: {statistics.median(latencies):.2f}ms")
    print(f"P95 Latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
    print(f"P99 Latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")

if __name__ == "__main__":
    run_load_test()
