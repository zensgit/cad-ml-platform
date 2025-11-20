#!/usr/bin/env python3
"""
Export evaluation metrics to Prometheus or OpenTelemetry.

Provides metrics in various formats for external monitoring systems.

Usage:
    python3 scripts/export_eval_metrics.py [--format prometheus|otel|json]

    # For Prometheus Pushgateway:
    python3 scripts/export_eval_metrics.py --format prometheus --push-gateway http://localhost:9091
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error


class MetricsExporter:
    """Export evaluation metrics to monitoring systems."""

    def __init__(self, history_dir: str = "reports/eval_history"):
        self.history_dir = Path(history_dir)
        self.latest_metrics = {}
        self.time_series = []

    def load_latest_evaluation(self) -> Optional[Dict]:
        """Load the most recent evaluation results."""
        json_files = sorted(self.history_dir.glob("*_combined.json"))

        if not json_files:
            return None

        latest_file = json_files[-1]

        try:
            with open(latest_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
            return None

    def load_time_series(self, limit: int = 100) -> List[Dict]:
        """Load time series of evaluations."""
        json_files = sorted(self.history_dir.glob("*_combined.json"))[-limit:]
        time_series = []

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    time_series.append(data)
            except Exception:
                continue

        return time_series

    def format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        latest = self.load_latest_evaluation()
        if not latest:
            return ""

        lines = []

        # Add help and type information
        lines.append("# HELP cad_ml_evaluation_score CAD ML Platform evaluation scores")
        lines.append("# TYPE cad_ml_evaluation_score gauge")

        # Extract scores (handle both formats)
        if "scores" in latest:
            combined = latest["scores"]["combined"]
            vision = latest["scores"]["vision"]["score"]
            ocr = latest["scores"]["ocr"]["normalized"]
            vision_metrics = latest["scores"]["vision"]["metrics"]
            ocr_metrics = latest["scores"]["ocr"]["metrics"]
        else:
            # Legacy format
            combined = latest["combined"]["combined_score"]
            vision = latest["combined"]["vision_score"]
            ocr = latest["combined"]["ocr_score"]
            vision_metrics = latest["vision_metrics"]
            ocr_metrics = latest["ocr_metrics"]

        lines.append(f'cad_ml_evaluation_score{{module="combined"}} {combined:.4f}')
        lines.append(f'cad_ml_evaluation_score{{module="vision"}} {vision:.4f}')
        lines.append(f'cad_ml_evaluation_score{{module="ocr"}} {ocr:.4f}')

        # Detailed metrics
        lines.append("# HELP cad_ml_vision_metrics Vision module detailed metrics")
        lines.append("# TYPE cad_ml_vision_metrics gauge")

        lines.append(f'cad_ml_vision_metrics{{metric="avg_hit_rate"}} {vision_metrics.get("AVG_HIT_RATE", 0):.4f}')
        lines.append(f'cad_ml_vision_metrics{{metric="min_hit_rate"}} {vision_metrics.get("MIN_HIT_RATE", 0):.4f}')
        lines.append(f'cad_ml_vision_metrics{{metric="max_hit_rate"}} {vision_metrics.get("MAX_HIT_RATE", 0):.4f}')
        lines.append(f'cad_ml_vision_metrics{{metric="num_samples"}} {vision_metrics.get("NUM_SAMPLES", 0)}')

        # OCR metrics
        lines.append("# HELP cad_ml_ocr_metrics OCR module detailed metrics")
        lines.append("# TYPE cad_ml_ocr_metrics gauge")

        lines.append(f'cad_ml_ocr_metrics{{metric="dimension_recall"}} {ocr_metrics.get("dimension_recall", 0):.4f}')
        lines.append(f'cad_ml_ocr_metrics{{metric="brier_score"}} {ocr_metrics.get("brier_score", 0):.4f}')
        lines.append(f'cad_ml_ocr_metrics{{metric="edge_f1"}} {ocr_metrics.get("edge_f1", 0):.4f}')

        # Metadata
        lines.append("# HELP cad_ml_evaluation_info Evaluation metadata")
        lines.append("# TYPE cad_ml_evaluation_info info")

        git_branch = latest.get("git_info", {}).get("branch", "unknown")
        git_commit = latest.get("git_info", {}).get("commit", "unknown")
        lines.append(f'cad_ml_evaluation_info{{branch="{git_branch}",commit="{git_commit}"}} 1')

        # Timestamp
        lines.append("# HELP cad_ml_evaluation_timestamp Last evaluation timestamp")
        lines.append("# TYPE cad_ml_evaluation_timestamp gauge")

        timestamp = datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00"))
        unix_timestamp = int(timestamp.timestamp())
        lines.append(f'cad_ml_evaluation_timestamp {unix_timestamp}')

        return "\n".join(lines)

    def format_opentelemetry(self) -> Dict:
        """Format metrics in OpenTelemetry JSON format."""
        latest = self.load_latest_evaluation()
        if not latest:
            return {}

        timestamp = datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00"))
        unix_nano = int(timestamp.timestamp() * 1e9)

        metrics = {
            "resource_metrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"string_value": "cad-ml-platform"}},
                        {"key": "service.version", "value": {"string_value": "1.0.0"}}
                    ]
                },
                "scope_metrics": [{
                    "scope": {
                        "name": "cad.ml.evaluation",
                        "version": "1.0.0"
                    },
                    "metrics": [
                        {
                            "name": "evaluation.score",
                            "description": "CAD ML evaluation scores",
                            "unit": "1",
                            "gauge": {
                                "data_points": [
                                    {
                                        "attributes": [{"key": "module", "value": {"string_value": "combined"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["combined"]
                                    },
                                    {
                                        "attributes": [{"key": "module", "value": {"string_value": "vision"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["vision"]["score"]
                                    },
                                    {
                                        "attributes": [{"key": "module", "value": {"string_value": "ocr"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["ocr"]["normalized"]
                                    }
                                ]
                            }
                        },
                        {
                            "name": "vision.hit_rate",
                            "description": "Vision module hit rates",
                            "unit": "1",
                            "gauge": {
                                "data_points": [
                                    {
                                        "attributes": [{"key": "type", "value": {"string_value": "avg"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["vision"]["metrics"]["AVG_HIT_RATE"]
                                    },
                                    {
                                        "attributes": [{"key": "type", "value": {"string_value": "min"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["vision"]["metrics"]["MIN_HIT_RATE"]
                                    },
                                    {
                                        "attributes": [{"key": "type", "value": {"string_value": "max"}}],
                                        "time_unix_nano": unix_nano,
                                        "as_double": latest["scores"]["vision"]["metrics"]["MAX_HIT_RATE"]
                                    }
                                ]
                            }
                        }
                    ]
                }]
            }]
        }

        return metrics

    def format_json_simple(self) -> Dict:
        """Format metrics in simple JSON format."""
        latest = self.load_latest_evaluation()
        if not latest:
            return {}

        return {
            "timestamp": latest["timestamp"],
            "metrics": {
                "combined_score": latest["scores"]["combined"],
                "vision_score": latest["scores"]["vision"]["score"],
                "ocr_score": latest["scores"]["ocr"]["normalized"],
                "vision": {
                    "avg_hit_rate": latest["scores"]["vision"]["metrics"]["AVG_HIT_RATE"],
                    "min_hit_rate": latest["scores"]["vision"]["metrics"]["MIN_HIT_RATE"],
                    "max_hit_rate": latest["scores"]["vision"]["metrics"]["MAX_HIT_RATE"],
                    "num_samples": latest["scores"]["vision"]["metrics"]["NUM_SAMPLES"]
                },
                "ocr": {
                    "dimension_recall": latest["scores"]["ocr"]["metrics"]["dimension_recall"],
                    "brier_score": latest["scores"]["ocr"]["metrics"]["brier_score"],
                    "edge_f1": latest["scores"]["ocr"]["metrics"].get("edge_f1", 0)
                }
            },
            "metadata": {
                "git_branch": latest.get("git_info", {}).get("branch", "unknown"),
                "git_commit": latest.get("git_info", {}).get("commit", "unknown"),
                "schema_version": latest.get("schema_version", "1.0.0")
            }
        }

    def push_to_prometheus_gateway(self, gateway_url: str, job_name: str = "cad_ml_eval") -> bool:
        """Push metrics to Prometheus Pushgateway."""
        metrics = self.format_prometheus()
        if not metrics:
            return False

        url = f"{gateway_url}/metrics/job/{job_name}"

        try:
            req = urllib.request.Request(
                url,
                data=metrics.encode('utf-8'),
                method='POST',
                headers={'Content-Type': 'text/plain'}
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() in [200, 202, 204]:
                    print(f"Successfully pushed metrics to {gateway_url}")
                    return True
                else:
                    print(f"Unexpected response code: {response.getcode()}")
                    return False

        except urllib.error.URLError as e:
            print(f"Failed to push metrics: {e}")
            return False
        except Exception as e:
            print(f"Error pushing metrics: {e}")
            return False

    def export_to_file(self, format_type: str, output_file: str) -> None:
        """Export metrics to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "prometheus":
            content = self.format_prometheus()
            with open(output_path, "w") as f:
                f.write(content)
        elif format_type == "otel":
            content = self.format_opentelemetry()
            with open(output_path, "w") as f:
                json.dump(content, f, indent=2)
        else:  # json
            content = self.format_json_simple()
            with open(output_path, "w") as f:
                json.dump(content, f, indent=2)

        print(f"Metrics exported to: {output_path}")

    def start_metrics_server(self, port: int = 8000) -> None:
        """Start a simple HTTP server for Prometheus scraping with graceful shutdown."""
        import http.server
        import socketserver
        import signal
        import threading

        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    exporter = MetricsExporter()
                    metrics = exporter.format_prometheus()

                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(metrics.encode('utf-8'))
                elif self.path == '/health':
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress log messages for normal requests
                if '/metrics' not in args[0] and '/health' not in args[0]:
                    super().log_message(format, *args)

        # Use TCPServer with SO_REUSEADDR to avoid "Address already in use" errors
        class ReuseAddrTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

            def server_close(self):
                """Enhanced cleanup on server close."""
                try:
                    self.shutdown()
                except:
                    pass
                super().server_close()

        server = None
        server_thread = None

        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully."""
            print("\n🛑 Received shutdown signal, cleaning up...")
            if server:
                # Stop accepting new connections
                server.shutdown()
                # Wait for ongoing requests to complete (with timeout)
                if server_thread and server_thread.is_alive():
                    server_thread.join(timeout=2.0)
                # Close the server socket
                server.server_close()
            print("✅ Metrics server stopped cleanly")
            sys.exit(0)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            server = ReuseAddrTCPServer(("", port), MetricsHandler)
            server.timeout = 1.0  # Allow periodic checks for shutdown

            print(f"📊 Metrics server running on port {port}")
            print(f"   Prometheus endpoint: http://localhost:{port}/metrics")
            print(f"   Health check: http://localhost:{port}/health")
            print("   Press Ctrl+C to stop gracefully")

            # Run server in a separate thread for better control
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = False  # Don't abruptly kill on main thread exit
            server_thread.start()

            # Keep main thread alive
            while server_thread.is_alive():
                server_thread.join(timeout=1.0)

        except OSError as e:
            if 'Address already in use' in str(e):
                print(f"❌ Port {port} is already in use. Try a different port with --port")
                print("   To find what's using the port: lsof -i :{port} or netstat -an | grep {port}")
            else:
                print(f"❌ Failed to start server: {e}")
            return
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if server:
                server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Export evaluation metrics")
    parser.add_argument("--format", choices=["prometheus", "otel", "json"],
                        default="prometheus", help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--push-gateway", help="Prometheus Pushgateway URL")
    parser.add_argument("--job-name", default="cad_ml_eval",
                        help="Job name for Pushgateway")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server for Prometheus scraping")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for metrics server")

    args = parser.parse_args()

    exporter = MetricsExporter()

    # Check if we have data
    latest = exporter.load_latest_evaluation()
    if not latest:
        print("No evaluation data found")
        return 1

    # Handle different modes
    if args.serve:
        exporter.start_metrics_server(port=args.port)
    elif args.push_gateway:
        success = exporter.push_to_prometheus_gateway(args.push_gateway, args.job_name)
        return 0 if success else 1
    elif args.output:
        exporter.export_to_file(args.format, args.output)
    else:
        # Output to stdout
        if args.format == "prometheus":
            print(exporter.format_prometheus())
        elif args.format == "otel":
            print(json.dumps(exporter.format_opentelemetry(), indent=2))
        else:
            print(json.dumps(exporter.format_json_simple(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())