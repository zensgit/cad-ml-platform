/**
 * CAD Assistant JavaScript/TypeScript SDK
 *
 * A client library for interacting with the CAD Assistant API.
 *
 * Usage:
 *   import { CADAssistant } from './cad-assistant';
 *
 *   const client = new CADAssistant({ apiKey: 'your-api-key' });
 *
 *   // Simple query
 *   const response = await client.ask('304不锈钢的抗拉强度是多少？');
 *   console.log(response.answer);
 *
 *   // Streaming query
 *   for await (const chunk of client.askStream('解释焊接工艺')) {
 *     process.stdout.write(chunk);
 *   }
 */

// ========== Types ==========

export interface CADAssistantConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  tenantId?: string;
}

export interface Source {
  type: string;
  content: string;
  relevance: number;
}

export interface AskResponse {
  answer: string;
  confidence: number;
  conversationId?: string;
  sources: Source[];
  modelUsed?: string;
  latencyMs: number;
}

export interface Conversation {
  id: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface KnowledgeResult {
  id: string;
  content: string;
  category: string;
  relevance: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  timestamp: number;
}

export interface AskOptions {
  conversationId?: string;
}

export interface SearchOptions {
  category?: string;
  limit?: number;
}

// ========== Errors ==========

export class CADAssistantError extends Error {
  code?: string;
  status?: number;

  constructor(message: string, code?: string, status?: number) {
    super(message);
    this.name = 'CADAssistantError';
    this.code = code;
    this.status = status;
  }
}

export class RateLimitError extends CADAssistantError {
  constructor(message: string, code?: string) {
    super(message, code, 429);
    this.name = 'RateLimitError';
  }
}

export class AuthenticationError extends CADAssistantError {
  constructor(message: string, code?: string) {
    super(message, code, 401);
    this.name = 'AuthenticationError';
  }
}

export class ValidationError extends CADAssistantError {
  constructor(message: string, code?: string) {
    super(message, code, 400);
    this.name = 'ValidationError';
  }
}

// ========== Client ==========

export class CADAssistant {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private tenantId?: string;
  private conversationId?: string;

  constructor(config: CADAssistantConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = (config.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.timeout = config.timeout || 30000;
    this.tenantId = config.tenantId;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'X-API-Key': this.apiKey,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    if (this.tenantId) {
      headers['X-Tenant-ID'] = this.tenantId;
    }
    return headers;
  }

  private async request<T>(
    method: string,
    path: string,
    data?: unknown
  ): Promise<T & { _latencyMs: number }> {
    const url = `${this.baseUrl}${path}`;
    const startTime = Date.now();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.getHeaders(),
        body: data ? JSON.stringify(data) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const latencyMs = Date.now() - startTime;

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const errorMsg = errorBody.error?.message || response.statusText;
        const errorCode = errorBody.error?.code || 'unknown';

        if (response.status === 401) {
          throw new AuthenticationError(errorMsg, errorCode);
        } else if (response.status === 429) {
          throw new RateLimitError(errorMsg, errorCode);
        } else if (response.status === 400) {
          throw new ValidationError(errorMsg, errorCode);
        } else {
          throw new CADAssistantError(errorMsg, errorCode, response.status);
        }
      }

      const result = await response.json();
      return { ...result, _latencyMs: latencyMs };
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof CADAssistantError) {
        throw error;
      }
      if (error instanceof Error && error.name === 'AbortError') {
        throw new CADAssistantError('Request timeout');
      }
      throw new CADAssistantError(String(error));
    }
  }

  // ========== Health ==========

  async health(): Promise<HealthStatus> {
    return this.request<HealthStatus>('GET', '/health');
  }

  // ========== Ask ==========

  async ask(query: string, options?: AskOptions): Promise<AskResponse> {
    const data: Record<string, unknown> = { query };
    const convId = options?.conversationId || this.conversationId;
    if (convId) {
      data.conversation_id = convId;
    }

    const result = await this.request<{ data: Record<string, unknown> }>('POST', '/ask', data);
    const responseData = result.data || {};

    const sources: Source[] = (responseData.sources as unknown[] || []).map((s: unknown) => {
      const source = s as Record<string, unknown>;
      return {
        type: String(source.type || 'unknown'),
        content: String(source.content || ''),
        relevance: Number(source.relevance || 0),
      };
    });

    const response: AskResponse = {
      answer: String(responseData.answer || ''),
      confidence: Number(responseData.confidence || 0),
      conversationId: responseData.conversation_id as string | undefined,
      sources,
      latencyMs: result._latencyMs,
    };

    if (response.conversationId) {
      this.conversationId = response.conversationId;
    }

    return response;
  }

  async *askStream(query: string, options?: AskOptions): AsyncGenerator<string> {
    const url = `${this.baseUrl}/ask/stream`;
    const data: Record<string, unknown> = { query };
    const convId = options?.conversationId || this.conversationId;
    if (convId) {
      data.conversation_id = convId;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        ...this.getHeaders(),
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new CADAssistantError(`Stream error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new CADAssistantError('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') {
              return;
            }
            try {
              const event = JSON.parse(dataStr);
              if (event.type === 'chunk') {
                yield event.data?.text || '';
              }
            } catch {
              continue;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ========== Conversations ==========

  async createConversation(): Promise<Conversation> {
    const result = await this.request<Record<string, unknown>>('POST', '/conversations');
    return {
      id: String(result.id || ''),
      createdAt: Number(result.created_at || 0),
      updatedAt: Number(result.updated_at || 0),
      messageCount: Number(result.message_count || 0),
    };
  }

  async listConversations(limit = 20, offset = 0): Promise<Conversation[]> {
    const result = await this.request<Record<string, unknown>[]>(
      'GET',
      `/conversations?limit=${limit}&offset=${offset}`
    );
    return result.map((c) => ({
      id: String(c.id || ''),
      createdAt: Number(c.created_at || 0),
      updatedAt: Number(c.updated_at || 0),
      messageCount: Number(c.message_count || 0),
    }));
  }

  async getConversation(conversationId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/conversations/${conversationId}`);
  }

  async deleteConversation(conversationId: string): Promise<boolean> {
    await this.request('DELETE', `/conversations/${conversationId}`);
    return true;
  }

  // ========== Knowledge ==========

  async searchKnowledge(query: string, options?: SearchOptions): Promise<KnowledgeResult[]> {
    const data: Record<string, unknown> = {
      query,
      limit: options?.limit || 10,
    };
    if (options?.category) {
      data.category = options.category;
    }

    const result = await this.request<{ results: Record<string, unknown>[] }>(
      'POST',
      '/knowledge/search',
      data
    );

    return (result.results || []).map((r) => ({
      id: String(r.id || ''),
      content: String(r.content || ''),
      category: String(r.category || ''),
      relevance: Number(r.relevance || 0),
    }));
  }

  // ========== Session Management ==========

  setConversationId(id: string): void {
    this.conversationId = id;
  }

  clearConversation(): void {
    this.conversationId = undefined;
  }
}

// ========== Factory Function ==========

export function createClient(config: CADAssistantConfig): CADAssistant {
  return new CADAssistant(config);
}

// Default export
export default CADAssistant;
