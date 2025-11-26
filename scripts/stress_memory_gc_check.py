import os
import time
import gc
import tracemalloc


def get_rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss)
    except Exception:
        # Fallback approximate via resource module (POSIX)
        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is kilobytes on Linux, bytes on macOS; normalize best-effort
            rss_kb = getattr(usage, "ru_maxrss", 0)
            return int(rss_kb) * 1024
        except Exception:
            return 0


def main(duration_seconds: float = 5.0) -> None:
    tracemalloc.start()
    gc.collect()
    base_rss = get_rss_bytes()

    blobs = []
    start = time.time()
    # Allocate and release to exercise GC
    while time.time() - start < duration_seconds:
        blobs.append(bytearray(1024 * 256))  # 256 KB
        if len(blobs) > 200:
            blobs = blobs[50:]
        time.sleep(0.005)

    gc.collect()
    final_rss = get_rss_bytes()

    strict = os.getenv("STRESS_STRICT", "0") == "1"
    if base_rss > 0 and final_rss > 0:
        growth = (final_rss - base_rss) / max(base_rss, 1)
        payload = {"base_rss": base_rss, "final_rss": final_rss, "growth_ratio": round(growth, 3)}
        print(payload)
        if strict:
            assert growth < 0.1, "RSS growth exceeds 10%"
    else:
        print({"base_rss": base_rss, "final_rss": final_rss, "growth_ratio": None, "note": "rss unavailable"})


if __name__ == "__main__":
    main()
