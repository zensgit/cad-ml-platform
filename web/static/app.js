/**
 * CAD Assistant Web UI
 * Main application JavaScript
 */

class CADAssistantUI {
    constructor() {
        this.apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8000';
        this.apiKey = localStorage.getItem('apiKey') || '';
        this.conversationId = null;
        this.isStreaming = false;

        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettings();
        this.checkApiStatus();
        this.loadAnalytics();

        // Check API status periodically
        setInterval(() => this.checkApiStatus(), 30000);
    }

    bindEvents() {
        // Navigation
        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.showPage(link.dataset.page);
            });
        });

        // Chat
        document.getElementById('send-btn').addEventListener('click', () => this.sendMessage());
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        document.getElementById('new-chat-btn').addEventListener('click', () => this.newConversation());

        // Quick queries
        document.querySelectorAll('.quick-query').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('chat-input').value = btn.textContent;
                this.sendMessage();
            });
        });

        // Settings
        document.getElementById('api-url').addEventListener('change', (e) => {
            this.apiUrl = e.target.value;
            localStorage.setItem('apiUrl', this.apiUrl);
        });
        document.getElementById('api-key').addEventListener('change', (e) => {
            this.apiKey = e.target.value;
            localStorage.setItem('apiKey', this.apiKey);
        });
    }

    showPage(pageName) {
        // Hide all pages
        document.querySelectorAll('[id$="-page"]').forEach(page => {
            page.classList.add('hidden');
        });

        // Show selected page
        const page = document.getElementById(`${pageName}-page`);
        if (page) {
            page.classList.remove('hidden');
        }

        // Update nav
        document.querySelectorAll('[data-page]').forEach(link => {
            link.classList.remove('bg-gray-700');
            if (link.dataset.page === pageName) {
                link.classList.add('bg-gray-700');
            }
        });

        // Update title
        const titles = {
            chat: ['智能对话', '与 CAD 助手交流，获取专业知识'],
            knowledge: ['知识库', '浏览和管理 CAD 知识库'],
            analytics: ['分析仪表板', '查看使用统计和质量趋势'],
            settings: ['设置', '配置 API 和模型选项']
        };

        if (titles[pageName]) {
            document.getElementById('page-title').textContent = titles[pageName][0];
            document.getElementById('page-subtitle').textContent = titles[pageName][1];
        }

        // Load page-specific data
        if (pageName === 'analytics') {
            this.loadAnalytics();
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const query = input.value.trim();

        if (!query) return;

        // Add user message
        this.addMessage('user', query);
        input.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await this.callApi('/ask', {
                query: query,
                conversation_id: this.conversationId
            });

            this.hideTypingIndicator();

            if (response.success) {
                this.conversationId = response.data.conversation_id;
                this.addMessage('assistant', response.data.answer, {
                    confidence: response.data.confidence,
                    sources: response.data.sources
                });
            } else {
                this.addMessage('error', response.error?.message || '请求失败');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('error', `错误: ${error.message}`);
        }
    }

    addMessage(role, content, metadata = {}) {
        const container = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');

        const isUser = role === 'user';
        const isError = role === 'error';

        messageDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;

        let bgClass = isUser ? 'message-user' : 'message-assistant';
        if (isError) bgClass = 'bg-red-100';

        let metaHtml = '';
        if (metadata.confidence) {
            const confidencePercent = Math.round(metadata.confidence * 100);
            metaHtml = `
                <div class="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                    置信度: ${confidencePercent}%
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="max-w-2xl ${bgClass} rounded-lg px-4 py-3 ${isError ? 'text-red-700' : ''}">
                <div class="flex items-center mb-1">
                    <i class="fas ${isUser ? 'fa-user' : isError ? 'fa-exclamation-circle' : 'fa-robot'} mr-2 text-gray-500"></i>
                    <span class="text-sm font-medium">${isUser ? '您' : isError ? '错误' : 'CAD 助手'}</span>
                </div>
                <div class="text-gray-800 whitespace-pre-wrap">${this.escapeHtml(content)}</div>
                ${metaHtml}
            </div>
        `;

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }

    showTypingIndicator() {
        const container = document.getElementById('chat-messages');
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'flex justify-start';
        indicator.innerHTML = `
            <div class="message-assistant rounded-lg px-4 py-3">
                <div class="typing-indicator flex space-x-1">
                    <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
                </div>
            </div>
        `;
        container.appendChild(indicator);
        container.scrollTop = container.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }

    newConversation() {
        this.conversationId = null;
        const container = document.getElementById('chat-messages');
        container.innerHTML = `
            <div class="flex justify-center">
                <div class="bg-white rounded-lg px-4 py-2 text-gray-500 text-sm">
                    开始新的对话
                </div>
            </div>
        `;
    }

    async checkApiStatus() {
        const statusEl = document.getElementById('api-status');
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            if (response.ok) {
                statusEl.innerHTML = '<i class="fas fa-circle text-xs mr-1"></i>在线';
                statusEl.className = 'flex items-center text-green-400';
            } else {
                throw new Error('API error');
            }
        } catch (error) {
            statusEl.innerHTML = '<i class="fas fa-circle text-xs mr-1"></i>离线';
            statusEl.className = 'flex items-center text-red-400';
        }
    }

    async loadAnalytics() {
        try {
            const response = await this.callApi('/metrics', null, 'GET');
            if (response.cache) {
                const cache = response.cache;

                // Update stats
                document.getElementById('stat-conversations').textContent =
                    cache.response?.size || 0;
                document.getElementById('stat-messages').textContent =
                    (cache.embedding?.hits || 0) + (cache.embedding?.misses || 0);

                const hitRate = cache.embedding?.hit_rate || 0;
                document.getElementById('stat-cache').textContent =
                    `${Math.round(hitRate * 100)}%`;
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }

        // Initialize chart (placeholder data)
        this.initQualityChart();
        this.loadTopics();
    }

    initQualityChart() {
        const ctx = document.getElementById('quality-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                datasets: [{
                    label: '质量分数',
                    data: [0.75, 0.78, 0.82, 0.80, 0.85, 0.83, 0.87],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.5,
                        max: 1.0
                    }
                }
            }
        });
    }

    loadTopics() {
        const topics = [
            { name: '材料属性', count: 45, color: 'blue' },
            { name: '焊接参数', count: 32, color: 'orange' },
            { name: 'GD&T', count: 28, color: 'green' },
            { name: '表面处理', count: 18, color: 'purple' },
            { name: '紧固件', count: 12, color: 'pink' }
        ];

        const container = document.getElementById('topics-list');
        if (!container) return;

        container.innerHTML = topics.map(topic => `
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <div class="w-3 h-3 rounded-full bg-${topic.color}-500 mr-3"></div>
                    <span>${topic.name}</span>
                </div>
                <span class="text-gray-500">${topic.count} 次查询</span>
            </div>
        `).join('');
    }

    loadSettings() {
        document.getElementById('api-url').value = this.apiUrl;
        document.getElementById('api-key').value = this.apiKey;
    }

    async callApi(endpoint, data, method = 'POST') {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (this.apiKey) {
            options.headers['X-API-Key'] = this.apiKey;
        }

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.apiUrl}${endpoint}`, options);
        return response.json();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CADAssistantUI();
});
