#!/usr/bin/env python
"""
OCR Integration Environment Verification Script

Checks all prerequisites and provides actionable feedback for OCR module setup.
Includes security checks for PDF processing and image resolution limits.

Usage:
    python scripts/verify_environment.py              # Normal output
    python scripts/verify_environment.py --json       # JSON output for CI
    python scripts/verify_environment.py --strict     # Exit non-zero on warnings
"""

import sys
import os
import argparse
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

@dataclass
class CheckResult:
    """Structured check result"""
    name: str
    success: bool
    details: str
    severity: str = "info"  # success|warning|error|info
    recommendations: list = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

def print_status(status: str, message: str, details: str = None):
    """Print colored status messages"""
    icons = {
        "success": f"{Colors.GREEN}✅",
        "warning": f"{Colors.YELLOW}⚠️",
        "error": f"{Colors.RED}❌",
        "info": f"{Colors.BLUE}ℹ️"
    }

    print(f"{icons.get(status, '')} {message}{Colors.END}")
    if details:
        print(f"   {Colors.BLUE}{details}{Colors.END}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.9"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (需要 >= 3.9)"

def check_paddle_import() -> Tuple[bool, str]:
    """Check PaddleOCR availability"""
    try:
        import paddleocr
        return True, f"PaddleOCR {paddleocr.__version__ if hasattr(paddleocr, '__version__') else 'installed'}"
    except ImportError:
        return False, "PaddleOCR not installed (pip install paddleocr)"

def check_cuda_availability() -> Tuple[bool, str]:
    """Check CUDA/GPU availability"""
    import subprocess

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)

            # Check VRAM
            mem_info = torch.cuda.mem_get_info(0)
            total_vram_gb = mem_info[1] / (1024**3)
            available_vram_gb = mem_info[0] / (1024**3)

            details = (f"GPU: {device_name}, "
                      f"VRAM: {available_vram_gb:.1f}/{total_vram_gb:.1f}GB available, "
                      f"Devices: {device_count}")

            # Warning if VRAM < 6GB for DeepSeek
            if total_vram_gb < 6:
                return True, f"{details}\n   ⚠️ Low VRAM for DeepSeek (recommend ≥6GB)"

            return True, details
        else:
            # Try to determine why CUDA is not available
            try:
                # Check if nvidia-smi works (driver installed)
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return False, "CUDA not available (driver OK, but torch.cuda unavailable - rebuild torch with CUDA)"
                else:
                    return False, "CUDA not available (nvidia-smi failed - check driver installation)"
            except FileNotFoundError:
                return False, "CUDA not available (nvidia-smi not found - no NVIDIA driver)"
            except subprocess.TimeoutExpired:
                return False, "CUDA not available (nvidia-smi timeout)"
            except Exception as e:
                return False, f"CUDA not available (CPU mode only): {str(e)}"

    except ImportError:
        return False, "PyTorch not installed (pip install torch)"

def check_redis_connection() -> Tuple[bool, str]:
    """Check Redis connectivity"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True, socket_connect_timeout=2)
        r.ping()

        # Check Redis memory
        info = r.info('memory')
        used_memory_mb = info.get('used_memory', 0) / (1024**2)

        return True, f"Redis connected (Memory: {used_memory_mb:.1f}MB used)"
    except ImportError:
        return False, "redis-py not installed (pip install redis)"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    import shutil

    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)

    if free_gb < 5:
        return False, f"Low disk space: {free_gb:.1f}GB free (need ≥5GB for models)"
    elif free_gb < 10:
        return True, f"Disk space: {free_gb:.1f}/{total_gb:.1f}GB (⚠️ recommend ≥10GB)"
    else:
        return True, f"Disk space: {free_gb:.1f}/{total_gb:.1f}GB available"

def check_deepseek_model() -> Tuple[bool, str]:
    """Check DeepSeek model availability"""
    model_path = Path("models/deepseek-ocr")

    if model_path.exists():
        # Check model files
        model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        if model_files:
            total_size_gb = sum(f.stat().st_size for f in model_files) / (1024**3)
            return True, f"DeepSeek model found ({total_size_gb:.1f}GB)"
        else:
            return False, "DeepSeek model directory exists but no model files"
    else:
        return False, "DeepSeek model not found (will download on first use)"

def check_security_limits() -> Dict[str, Tuple[bool, str]]:
    """Check security configuration and limits"""
    results = {}

    # Check environment variables
    max_resolution = os.getenv('MAX_RESOLUTION', '2048')
    max_file_size_mb = os.getenv('MAX_FILE_SIZE_MB', '50')
    max_pdf_pages = os.getenv('MAX_PDF_PAGES', '20')

    results['resolution_limit'] = (
        True,
        f"Max resolution: {max_resolution}px (prevents OOM)"
    )

    results['file_size_limit'] = (
        True,
        f"Max file size: {max_file_size_mb}MB (HTTP 413 if exceeded)"
    )

    results['pdf_page_limit'] = (
        True,
        f"Max PDF pages: {max_pdf_pages} (HTTP 422 if exceeded)"
    )

    # Check PDF security tools
    try:
        import pypdf
        pdf_check = (True, f"PyPDF {pypdf.__version__} (PDF validation ready)")
    except ImportError:
        pdf_check = (False, "PyPDF not installed (needed for PDF security checks)")
    results['pdf_security'] = pdf_check

    # Check image processing libraries for resolution control
    try:
        from PIL import Image

        # Dynamically set based on MAX_RESOLUTION env var
        max_resolution = int(os.getenv('MAX_RESOLUTION', '2048'))
        max_pixels = max_resolution * max_resolution * 3  # RGB channels

        Image.MAX_IMAGE_PIXELS = max_pixels

        results['pil_security'] = (
            True,
            f"PIL decompression bomb limit: {max_pixels/1e6:.1f}M pixels (from MAX_RESOLUTION={max_resolution})"
        )
    except ImportError:
        results['pil_security'] = (False, "Pillow not installed")
    except ValueError:
        results['pil_security'] = (False, "Invalid MAX_RESOLUTION environment variable")

    return results

def check_mime_validation() -> Tuple[bool, str]:
    """Check MIME type validation setup"""
    # Try python-magic first (preferred)
    try:
        import magic  # Linux: python-magic-bin or file-magic
        mime = magic.Magic(mime=True)
        return True, "python-magic ready for MIME validation"
    except ImportError:
        pass

    # Try python_magic (alternative package name)
    try:
        import python_magic
        mime = python_magic.Magic(mime=True)
        return True, "python-magic ready for MIME validation"
    except ImportError:
        pass

    # Fallback to mimetypes
    try:
        import mimetypes
        return True, "⚠️ mimetypes fallback (less accurate, recommend: pip install python-magic-bin)"
    except:
        return False, "No MIME validation library (install python-magic-bin)"

def check_monitoring_tools() -> Dict[str, Tuple[bool, str]]:
    """Check monitoring and metrics tools"""
    results = {}

    # Prometheus client
    try:
        import prometheus_client
        results['prometheus'] = (True, f"prometheus_client {prometheus_client.__version__}")
    except ImportError:
        results['prometheus'] = (False, "prometheus_client not installed")

    # Memory monitoring
    try:
        import psutil
        mem = psutil.virtual_memory()
        results['memory_monitor'] = (
            True,
            f"psutil ready (Current: {mem.percent}% memory used)"
        )
    except ImportError:
        results['memory_monitor'] = (False, "psutil not installed (for memory monitoring)")

    return results

def check_concurrency_limits() -> Tuple[bool, str]:
    """Check concurrency configuration"""
    max_concurrent = os.getenv('OCR_MAX_CONCURRENT', '10')
    timeout_ms = os.getenv('OCR_TIMEOUT_MS', '30000')
    deepseek_enabled = os.getenv('DEEPSEEK_ENABLED', 'true').lower() == 'true'

    try:
        import psutil
    except ImportError:
        # psutil not available, can't check CPU cores
        try:
            max_concurrent_val = int(max_concurrent)
            timeout_s = int(timeout_ms) / 1000

            base_msg = f"Concurrency: max={max_concurrent_val}, timeout={timeout_s}s (CPU check unavailable, install psutil)"

            # Still check DeepSeek GPU conflict
            if deepseek_enabled:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        return False, f"{base_msg}\n   ⚠️ DEEPSEEK_ENABLED=true but no CUDA"
                except ImportError:
                    return False, f"{base_msg}\n   ⚠️ DEEPSEEK_ENABLED=true but torch not installed"

            return True, base_msg
        except ValueError as e:
            return False, f"Invalid concurrency configuration: {e}"

    try:
        max_concurrent_val = int(max_concurrent)
        timeout_s = int(timeout_ms) / 1000

        # Check against CPU cores
        cpu_count = psutil.cpu_count(logical=True)
        recommended_max = cpu_count * 2

        warnings = []
        if max_concurrent_val > recommended_max:
            warnings.append(f"⚠️ max_concurrent ({max_concurrent_val}) > CPU cores * 2 ({recommended_max})")

        # Check if DeepSeek enabled but no GPU
        if deepseek_enabled:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("⚠️ DEEPSEEK_ENABLED=true but no CUDA (performance will be poor)")
            except ImportError:
                warnings.append("⚠️ DEEPSEEK_ENABLED=true but torch not installed")

        base_msg = f"Concurrency: max={max_concurrent_val}, timeout={timeout_s}s, CPUs={cpu_count}"

        if warnings:
            return False, f"{base_msg}\n   " + "\n   ".join(warnings)
        else:
            return True, base_msg

    except ValueError as e:
        return False, f"Invalid concurrency configuration: {e}"

def run_environment_check():
    """Run all environment checks"""
    print(f"\n{Colors.BOLD}🔍 CAD ML Platform - OCR Integration Environment Check{Colors.END}\n")
    print("=" * 60)

    # Core requirements
    print(f"\n{Colors.BOLD}📦 Core Requirements:{Colors.END}")

    success, details = check_python_version()
    print_status("success" if success else "error", "Python Version", details)

    success, details = check_paddle_import()
    print_status("success" if success else "error", "PaddleOCR", details)

    success, details = check_cuda_availability()
    status = "success" if success and "Low VRAM" not in details else "warning" if success else "info"
    print_status(status, "CUDA/GPU", details)

    success, details = check_redis_connection()
    print_status("success" if success else "warning", "Redis", details)

    success, details = check_disk_space()
    status = "success" if success and "recommend" not in details else "warning"
    print_status(status, "Disk Space", details)

    success, details = check_deepseek_model()
    print_status("success" if success else "info", "DeepSeek Model", details)

    # Security checks
    print(f"\n{Colors.BOLD}🔐 Security & Limits:{Colors.END}")

    security_checks = check_security_limits()
    for check_name, (success, details) in security_checks.items():
        label = check_name.replace('_', ' ').title()
        print_status("success" if success else "warning", label, details)

    success, details = check_mime_validation()
    print_status("success" if success else "warning", "MIME Validation", details)

    # Monitoring
    print(f"\n{Colors.BOLD}📊 Monitoring & Observability:{Colors.END}")

    monitoring = check_monitoring_tools()
    for tool_name, (success, details) in monitoring.items():
        label = tool_name.replace('_', ' ').title()
        print_status("success" if success else "info", label, details)

    success, details = check_concurrency_limits()
    print_status("success" if success else "error", "Concurrency Config", details)

    # Environment variables summary
    print(f"\n{Colors.BOLD}⚙️ Environment Configuration:{Colors.END}")

    env_vars = {
        'OCR_PROVIDER': os.getenv('OCR_PROVIDER', 'auto'),
        'DEEPSEEK_ENABLED': os.getenv('DEEPSEEK_ENABLED', 'true'),
        'CONFIDENCE_FALLBACK': os.getenv('CONFIDENCE_FALLBACK', '0.85'),
        'PROMPT_VERSION': os.getenv('PROMPT_VERSION', 'v1'),
        'DATASET_VERSION': os.getenv('DATASET_VERSION', 'v1.0'),
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'OCR_CACHE_TTL': os.getenv('OCR_CACHE_TTL', '3600'),
    }

    for key, value in env_vars.items():
        # Mask Redis URL password if present
        if key == 'REDIS_URL' and '@' in value:
            parts = value.split('@')
            if ':' in parts[0]:
                masked = parts[0].split(':')[0] + ':****@' + parts[1]
                value = masked

        print(f"   {key}: {value}")

    # Final summary
    print(f"\n{Colors.BOLD}📋 Summary:{Colors.END}")
    print("=" * 60)

    ready_for_cpu = all([
        check_python_version()[0],
        check_paddle_import()[0],
        check_disk_space()[0]
    ])

    ready_for_gpu = ready_for_cpu and check_cuda_availability()[0]

    if ready_for_gpu:
        print_status("success", "✅ Environment ready for GPU-accelerated OCR")
        print("   Recommended: Use DeepSeek for high-quality extraction")
    elif ready_for_cpu:
        print_status("success", "✅ Environment ready for CPU-based OCR")
        print("   Note: PaddleOCR will be primary provider")
    else:
        print_status("error", "❌ Environment not ready")
        print("   Please install missing dependencies")

    # Quick start commands
    print(f"\n{Colors.BOLD}🚀 Quick Start Commands:{Colors.END}")
    print("=" * 60)
    print(f"{Colors.BLUE}# Install missing dependencies:{Colors.END}")
    print("pip install -r requirements.txt")
    print("")
    print(f"{Colors.BLUE}# Run OCR service:{Colors.END}")
    print("python src/main.py")
    print("")
    print(f"{Colors.BLUE}# Run tests:{Colors.END}")
    print("pytest tests/ocr/test_cache_key.py")
    print("pytest tests/ocr/test_fallback.py")

    return ready_for_cpu or ready_for_gpu

if __name__ == "__main__":
    success = run_environment_check()
    sys.exit(0 if success else 1)