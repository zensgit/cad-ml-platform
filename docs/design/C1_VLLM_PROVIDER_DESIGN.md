# C1: vLLM Provider Design

## Overview

Phase C1 adds a `VLLMProvider` to the CAD-ML Platform's multi-provider LLM
architecture. The provider connects to a locally-deployed vLLM server via its
OpenAI-compatible HTTP API, targeting sub-100ms inference latency for CAD domain
queries.

## Architecture

```
                         CAD-ML Platform
  +----------------------------------------------------------+
  |                                                          |
  |  CADAssistant                                            |
  |    |                                                     |
  |    +-- get_best_available_provider()                     |
  |         |                                                |
  |         +-- ClaudeProvider   (external API)              |
  |         +-- OpenAIProvider   (external API)              |
  |         +-- QwenProvider     (external API)              |
  |         +-- VLLMProvider     (local HTTP) <-- NEW        |
  |         +-- OllamaProvider   (local HTTP)                |
  |         +-- OfflineProvider  (no LLM)                    |
  |                                                          |
  +---------------------------+------------------------------+
                              |
                    HTTP (localhost:8100)
                              |
                    +---------v---------+
                    |   vLLM Server     |
                    |   (GPU container) |
                    |                   |
                    |  DeepSeek-Coder   |
                    |  6.7B AWQ         |
                    +-------------------+
```

### Provider Priority Order

```
Claude > OpenAI > Qwen > vLLM > Ollama > Offline
```

When `auto_select_provider` is enabled, the system checks each provider in order
and uses the first available one. vLLM sits above Ollama in priority because it
offers better throughput and supports GPU-accelerated quantized models.

## API Contract

VLLMProvider communicates with vLLM using the **OpenAI-compatible API**
(`/v1/chat/completions`).

### Request (non-streaming)

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "deepseek-coder-6.7b-awq",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.3,
  "max_tokens": 2000,
  "stream": false
}
```

### Response (non-streaming)

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

### Streaming (SSE)

Set `"stream": true`. The server returns `text/event-stream` with lines:

```
data: {"choices":[{"delta":{"content":"token"}}]}
data: {"choices":[{"delta":{"content":"s"}}]}
data: [DONE]
```

### Health Endpoints

| Endpoint      | Purpose                     |
|---------------|-----------------------------|
| GET /health   | vLLM native health check    |
| GET /v1/models| List loaded models (OpenAI) |

## Configuration Reference

| Environment Variable | Default                        | Description                              |
|---------------------|--------------------------------|------------------------------------------|
| VLLM_ENDPOINT       | http://localhost:8100           | vLLM server base URL                     |
| VLLM_MODEL          | deepseek-coder-6.7b-awq        | Model name as served by vLLM             |
| VLLM_TIMEOUT        | 30                             | HTTP request timeout (seconds)           |

The provider also respects standard `LLMConfig` fields (`temperature`,
`max_tokens`) passed through the `AssistantConfig` or `get_provider()`.

## Deployment Topology

```yaml
# docker-compose.yml (with --profile gpu)
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports: ["8100:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles: [gpu]
```

The vLLM service uses a Docker Compose **profile** (`gpu`) so it is only started
on hosts with NVIDIA GPUs:

```bash
# GPU host
docker compose --profile gpu up -d

# CPU-only host (vLLM is not started)
docker compose up -d
```

### Resource Requirements

| Resource | Minimum      | Recommended   |
|----------|-------------|---------------|
| GPU VRAM | 6 GB        | 8 GB          |
| RAM      | 8 GB        | 16 GB         |
| Disk     | 15 GB       | 30 GB         |
| GPU      | NVIDIA T4   | NVIDIA A10/L4 |

## Model Selection Rationale

**Recommended: DeepSeek-Coder-6.7B-Instruct AWQ**

| Criterion           | DeepSeek-Coder-6.7B AWQ |
|---------------------|-------------------------|
| Parameter count     | 6.7B (4-bit quantized)  |
| VRAM usage          | ~4 GB                   |
| Domain fit          | Code + engineering text  |
| Quantization method | AWQ (activation-aware)   |
| Inference speed     | 50-80 tok/s on T4       |
| License             | Permissive              |

AWQ quantization was chosen over GPTQ based on benchmark results showing 15-20%
better throughput at equivalent perplexity. The existing benchmark script at
`scripts/benchmark_vllm_quantization.py` can be used to validate on target
hardware.

## Performance Targets

| Metric                    | Target    | Measurement Method              |
|---------------------------|-----------|----------------------------------|
| Time to first token (P50) | <50ms     | benchmark_vllm_quantization.py   |
| End-to-end latency (P95)  | <100ms    | Prometheus histogram             |
| Throughput                 | >50 tok/s | benchmark_vllm_quantization.py   |
| Availability              | 99.5%     | Health check monitoring          |

## Fallback and Degradation Strategy

```
VLLMProvider.is_available() == False
        |
        v
get_best_available_provider() skips vLLM
        |
        v
Tries next: OllamaProvider
        |
        v
Last resort: OfflineProvider (knowledge-base only)
```

1. **Connection refused**: vLLM container not running. Provider returns
   `is_available() = False` within 2s timeout. System falls through to next
   provider.

2. **Slow response**: If vLLM exceeds `VLLM_TIMEOUT`, a `requests.Timeout`
   exception is raised. The `CADAssistant._call_llm()` catch-all handles this
   and falls back to the default offline callback.

3. **Model not loaded**: `/v1/models` returns empty list. `health_check()`
   reports this for diagnostics but `is_available()` still returns True if the
   server responds (the model may be loading).

4. **Feature flag disabled**: When `vllm_enabled` is `false` in
   `config/feature_flags.json`, the deployment should not route traffic to vLLM.
   This is enforced at the application layer by checking the flag before
   selecting the provider.

## Security Considerations

- vLLM binds only to the internal Docker network (`cad-ml-network`). Port 8100
  is exposed on the host for development; production deployments should restrict
  this with firewall rules or remove the port mapping.
- No authentication is required between the platform and vLLM since they
  communicate over a trusted internal network.
- Model weights are cached in a named Docker volume (`vllm-models`). Ensure the
  host filesystem permissions prevent unauthorized access.
- The `HUGGING_FACE_HUB_TOKEN` env var is passed through for gated model
  downloads and should be stored as a Docker secret or in a `.env` file excluded
  from version control.

## Migration Path

### Phase 1: Shadow mode (current)
- Deploy vLLM alongside existing providers
- Feature flag `vllm_enabled` = false
- Run benchmarks with `scripts/benchmark_vllm_quantization.py`
- Compare quality against Claude/OpenAI baseline

### Phase 2: Canary rollout
- Set `vllm_enabled` strategy to `percentage: 10`
- Monitor latency, error rate, and response quality
- Gradually increase percentage

### Phase 3: Default provider
- Set `vllm_enabled` to `true` with strategy `all`
- Update `get_best_available_provider` priority if vLLM should be preferred
  over external APIs for latency-sensitive paths
- Keep external API providers as fallback

### Phase 4: Optimization
- Tune `gpu_memory_utilization`, `max_model_len` based on production workload
- Evaluate larger models (13B, 34B) if GPU budget allows
- Add model-specific prompt templates for CAD domain
