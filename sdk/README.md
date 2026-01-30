# CAD Assistant SDK

Client libraries for the CAD Assistant API.

## Python SDK

### Installation

```bash
pip install cad-assistant
# or copy sdk/python/cad_assistant.py to your project
```

### Quick Start

```python
from cad_assistant import CADAssistant

# Initialize client
client = CADAssistant(api_key="your-api-key")

# Simple query
response = client.ask("304不锈钢的抗拉强度是多少？")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")

# Streaming query
for chunk in client.ask_stream("解释TIG焊接工艺"):
    print(chunk, end="", flush=True)

# Multi-turn conversation
with client.conversation() as conv:
    r1 = conv.ask("304不锈钢的密度是多少？")
    r2 = conv.ask("它的耐腐蚀性如何？")  # 自动关联上下文

# Knowledge search
results = client.search_knowledge("焊接参数", category="welding", limit=5)
for r in results:
    print(f"[{r.relevance:.2f}] {r.content}")
```

### Error Handling

```python
from cad_assistant import (
    CADAssistant,
    RateLimitError,
    AuthenticationError,
    ValidationError,
)

client = CADAssistant(api_key="your-api-key")

try:
    response = client.ask("query")
except RateLimitError:
    print("Rate limit exceeded, please wait")
except AuthenticationError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
```

## JavaScript/TypeScript SDK

### Installation

```bash
npm install cad-assistant
# or copy sdk/javascript/cad-assistant.ts to your project
```

### Quick Start

```typescript
import { CADAssistant } from './cad-assistant';

// Initialize client
const client = new CADAssistant({ apiKey: 'your-api-key' });

// Simple query
const response = await client.ask('304不锈钢的抗拉强度是多少？');
console.log(`Answer: ${response.answer}`);

// Streaming query
for await (const chunk of client.askStream('解释焊接工艺')) {
  process.stdout.write(chunk);
}

// Knowledge search
const results = await client.searchKnowledge('焊接参数', {
  category: 'welding',
  limit: 5,
});
```

### Browser Usage

```html
<script type="module">
import { CADAssistant } from './cad-assistant.js';

const client = new CADAssistant({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.example.com',
});

const response = await client.ask('材料查询');
document.getElementById('answer').textContent = response.answer;
</script>
```

## API Reference

### Common Methods

| Method | Description |
|--------|-------------|
| `ask(query, options?)` | Ask a question |
| `askStream(query, options?)` | Streaming response |
| `createConversation()` | Create new conversation |
| `listConversations(limit?, offset?)` | List conversations |
| `getConversation(id)` | Get conversation details |
| `deleteConversation(id)` | Delete conversation |
| `searchKnowledge(query, options?)` | Search knowledge base |
| `health()` | Check API health |

### Response Types

```typescript
interface AskResponse {
  answer: string;
  confidence: number;
  conversationId?: string;
  sources: Source[];
  latencyMs: number;
}

interface Source {
  type: string;
  content: string;
  relevance: number;
}
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiKey` | string | required | Your API key |
| `baseUrl` | string | `http://localhost:8000` | API base URL |
| `timeout` | number | 30000 | Request timeout (ms) |
| `tenantId` | string | - | Multi-tenant ID |
