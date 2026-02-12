"""
Conversation History Manager.

Manages multi-turn conversation context for the CAD assistant,
enabling context-aware responses across conversation turns.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from collections import deque


class MessageRole(str, Enum):
    """Role of a message in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in conversation history."""

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """Context extracted from conversation history."""

    # Extracted entities across turns
    materials: List[str] = field(default_factory=list)
    dimensions: List[float] = field(default_factory=list)
    tolerances: List[str] = field(default_factory=list)
    thread_specs: List[str] = field(default_factory=list)

    # Extended entities for new knowledge domains
    welding_processes: List[str] = field(default_factory=list)
    heat_treatments: List[str] = field(default_factory=list)
    surface_treatments: List[str] = field(default_factory=list)
    gdt_characteristics: List[str] = field(default_factory=list)

    # Topic tracking
    current_topic: Optional[str] = None
    topic_history: List[str] = field(default_factory=list)

    # Follow-up tracking
    last_query_intent: Optional[str] = None
    pending_clarifications: List[str] = field(default_factory=list)

    # Reference tracking for pronoun resolution
    last_mentioned_material: Optional[str] = None
    last_mentioned_dimension: Optional[float] = None
    last_mentioned_process: Optional[str] = None

    def merge(self, other: "ConversationContext") -> None:
        """Merge another context into this one."""
        self.materials.extend(other.materials)
        self.dimensions.extend(other.dimensions)
        self.tolerances.extend(other.tolerances)
        self.thread_specs.extend(other.thread_specs)
        self.welding_processes.extend(other.welding_processes)
        self.heat_treatments.extend(other.heat_treatments)
        self.surface_treatments.extend(other.surface_treatments)
        self.gdt_characteristics.extend(other.gdt_characteristics)

        if other.current_topic:
            self.current_topic = other.current_topic
            if other.current_topic not in self.topic_history:
                self.topic_history.append(other.current_topic)

        self.last_query_intent = other.last_query_intent or self.last_query_intent

        # Update reference tracking
        if other.last_mentioned_material:
            self.last_mentioned_material = other.last_mentioned_material
        if other.last_mentioned_dimension:
            self.last_mentioned_dimension = other.last_mentioned_dimension
        if other.last_mentioned_process:
            self.last_mentioned_process = other.last_mentioned_process


@dataclass
class Conversation:
    """A single conversation session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **metadata) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )
        self.messages.append(message)
        self.updated_at = time.time()
        return message

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages."""
        return self.messages[-n:]

    def get_messages_since(self, timestamp: float) -> List[Message]:
        """Get messages since a timestamp."""
        return [m for m in self.messages if m.timestamp >= timestamp]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        conv = cls(
            id=data["id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return conv


class ConversationManager:
    """
    Manages conversation history and context.

    Features:
    - Multi-turn context tracking
    - Entity extraction across turns
    - Topic continuity detection
    - Memory-efficient storage with limits

    Example:
        >>> manager = ConversationManager()
        >>> conv_id = manager.create_conversation()
        >>> manager.add_user_message(conv_id, "304不锈钢的强度是多少？")
        >>> context = manager.get_context(conv_id)
    """

    def __init__(
        self,
        max_conversations: int = 100,
        max_messages_per_conversation: int = 50,
        context_window: int = 10,
    ):
        """
        Initialize conversation manager.

        Args:
            max_conversations: Maximum conversations to keep in memory
            max_messages_per_conversation: Max messages per conversation
            context_window: Number of recent messages for context building
        """
        self.max_conversations = max_conversations
        self.max_messages = max_messages_per_conversation
        self.context_window = context_window

        self._conversations: Dict[str, Conversation] = {}
        self._conversation_order: deque = deque(maxlen=max_conversations)

    def create_conversation(self, **metadata) -> str:
        """
        Create a new conversation.

        Returns:
            Conversation ID
        """
        conv = Conversation(metadata=metadata)

        # Evict oldest if at capacity
        if len(self._conversations) >= self.max_conversations:
            oldest_id = self._conversation_order.popleft()
            self._conversations.pop(oldest_id, None)

        self._conversations[conv.id] = conv
        self._conversation_order.append(conv.id)

        return conv.id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        **metadata,
    ) -> Optional[Message]:
        """Add a user message to a conversation."""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None

        message = conv.add_message(MessageRole.USER, content, **metadata)

        # Trim if too many messages
        if len(conv.messages) > self.max_messages:
            conv.messages = conv.messages[-self.max_messages:]

        # Update context
        self._update_context(conv, content)

        return message

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        **metadata,
    ) -> Optional[Message]:
        """Add an assistant message to a conversation."""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None

        message = conv.add_message(MessageRole.ASSISTANT, content, **metadata)

        # Trim if too many messages
        if len(conv.messages) > self.max_messages:
            conv.messages = conv.messages[-self.max_messages:]

        return message

    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get the conversation context."""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None
        return conv.context

    def get_context_summary(self, conversation_id: str) -> str:
        """
        Get a text summary of conversation context.

        Useful for including in LLM prompts.
        """
        conv = self._conversations.get(conversation_id)
        if not conv:
            return ""

        ctx = conv.context
        parts = []

        if ctx.materials:
            parts.append(f"讨论的材料: {', '.join(set(ctx.materials[-5:]))}")

        if ctx.dimensions:
            parts.append(f"提到的尺寸: {', '.join(str(d) for d in set(ctx.dimensions[-5:]))}mm")

        if ctx.tolerances:
            parts.append(f"公差相关: {', '.join(set(ctx.tolerances[-3:]))}")

        if ctx.thread_specs:
            parts.append(f"螺纹规格: {', '.join(set(ctx.thread_specs[-3:]))}")

        # Extended context summaries
        if ctx.welding_processes:
            parts.append(f"焊接工艺: {', '.join(set(ctx.welding_processes[-3:]))}")

        if ctx.heat_treatments:
            parts.append(f"热处理: {', '.join(set(ctx.heat_treatments[-3:]))}")

        if ctx.surface_treatments:
            parts.append(f"表面处理: {', '.join(set(ctx.surface_treatments[-3:]))}")

        if ctx.gdt_characteristics:
            parts.append(f"几何公差: {', '.join(set(ctx.gdt_characteristics[-3:]))}")

        if ctx.current_topic:
            parts.append(f"当前话题: {ctx.current_topic}")

        return "; ".join(parts)

    def get_recent_history(
        self,
        conversation_id: str,
        n: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for LLM context.

        Returns list of {"role": "user/assistant", "content": "..."}
        """
        conv = self._conversations.get(conversation_id)
        if not conv:
            return []

        recent = conv.get_recent_messages(n)
        return [{"role": m.role.value, "content": m.content} for m in recent]

    def build_conversation_prompt(
        self,
        conversation_id: str,
        current_query: str,
        max_history: int = 3,
    ) -> str:
        """
        Build a prompt that includes conversation history.

        Args:
            conversation_id: Conversation ID
            current_query: Current user query
            max_history: Max previous turns to include

        Returns:
            Prompt string with history and current query
        """
        conv = self._conversations.get(conversation_id)
        if not conv:
            return current_query

        # Get recent history
        history = self.get_recent_history(conversation_id, max_history * 2)

        # Build prompt
        parts = []

        if history:
            parts.append("之前的对话:")
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                parts.append(f"{role}: {msg['content']}")
            parts.append("")

        # Add context summary
        context_summary = self.get_context_summary(conversation_id)
        if context_summary:
            parts.append(f"对话背景: {context_summary}")
            parts.append("")

        parts.append(f"当前问题: {current_query}")

        return "\n".join(parts)

    def detect_follow_up(self, conversation_id: str, query: str) -> bool:
        """
        Detect if a query is a follow-up to previous context.

        Returns True if query seems to reference previous conversation.
        """
        conv = self._conversations.get(conversation_id)
        if not conv or len(conv.messages) < 2:
            return False

        # Check for pronouns and references
        follow_up_indicators = [
            "它", "这个", "那个", "上面", "刚才",
            "这种", "那种", "其", "该",
            "还有", "另外", "还需要", "再问",
            "那么", "所以", "如果",
        ]

        query_lower = query.lower()
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True

        # Check if query is very short (likely follow-up)
        if len(query) < 10:
            return True

        return False

    def resolve_references(self, conversation_id: str, query: str) -> str:
        """
        Resolve pronoun references using conversation context.

        Example: "它的密度是多少？" -> "304不锈钢的密度是多少？"
        """
        conv = self._conversations.get(conversation_id)
        if not conv:
            return query

        ctx = conv.context
        resolved_query = query

        # Material reference patterns
        material_refs = [
            "它的", "这个材料的", "该材料的", "这种钢的",
            "这种材料的", "那个材料的", "上面材料的",
        ]

        # Process reference patterns
        process_refs = [
            "这个工艺的", "该工艺的", "这种处理的",
            "那个处理的", "上面工艺的", "刚才说的工艺",
        ]

        # Dimension reference patterns
        dimension_refs = [
            "这个尺寸的", "该尺寸的", "这个直径的",
            "那个直径的", "刚才那个尺寸",
        ]

        # Try to resolve material references
        if ctx.last_mentioned_material:
            for ref in material_refs:
                if ref in resolved_query:
                    resolved_query = resolved_query.replace(
                        ref, f"{ctx.last_mentioned_material}的"
                    )
                    break

        # Try to resolve process references
        if ctx.last_mentioned_process:
            for ref in process_refs:
                if ref in resolved_query:
                    resolved_query = resolved_query.replace(
                        ref, f"{ctx.last_mentioned_process}的"
                    )
                    break

        # Try to resolve dimension references
        if ctx.last_mentioned_dimension:
            for ref in dimension_refs:
                if ref in resolved_query:
                    resolved_query = resolved_query.replace(
                        ref, f"{ctx.last_mentioned_dimension}mm的"
                    )
                    break

        # Fallback: try materials list if last_mentioned_material not set
        if resolved_query == query and ctx.materials:
            last_material = ctx.materials[-1]
            for ref in material_refs:
                if ref in resolved_query:
                    resolved_query = resolved_query.replace(ref, f"{last_material}的")
                    break

        return resolved_query

    def _update_context(self, conv: Conversation, content: str) -> None:
        """Update conversation context from new content."""
        import re
        ctx = conv.context

        # Extract materials (common patterns)
        material_patterns = [
            "304", "316", "316L", "Q235", "45#", "45钢", "40Cr",
            "6061", "7075", "TC4", "TA2", "H62", "H65",
            "不锈钢", "碳钢", "铝合金", "钛合金", "黄铜",
        ]

        for pattern in material_patterns:
            if pattern in content:
                ctx.materials.append(pattern)
                ctx.last_mentioned_material = pattern

        # Extract dimensions (numbers followed by mm)
        dim_matches = re.findall(r'(\d+(?:\.\d+)?)\s*mm', content)
        for match in dim_matches:
            try:
                dim_value = float(match)
                ctx.dimensions.append(dim_value)
                ctx.last_mentioned_dimension = dim_value
            except ValueError:
                pass

        # Extract thread specs
        thread_matches = re.findall(r'M\d+(?:x\d+(?:\.\d+)?)?', content, re.IGNORECASE)
        ctx.thread_specs.extend(thread_matches)

        # Extract tolerance references
        tol_patterns = ["IT", "H7", "g6", "H8", "f7", "公差"]
        for pattern in tol_patterns:
            if pattern in content:
                ctx.tolerances.append(pattern)

        # Extract welding processes
        welding_patterns = [
            "TIG", "MIG", "MAG", "SMAW", "GTAW", "GMAW",
            "氩弧焊", "埋弧焊", "电弧焊", "激光焊", "点焊", "缝焊",
            "焊接", "焊缝", "焊丝", "焊条",
        ]
        for pattern in welding_patterns:
            if pattern in content:
                ctx.welding_processes.append(pattern)
                ctx.last_mentioned_process = pattern

        # Extract heat treatments
        heat_treatment_patterns = [
            "淬火", "回火", "退火", "正火", "调质", "渗碳", "渗氮",
            "时效", "固溶", "HRC", "HB", "HV",
            "quenching", "tempering", "annealing", "normalizing",
        ]
        for pattern in heat_treatment_patterns:
            if pattern in content:
                ctx.heat_treatments.append(pattern)
                ctx.last_mentioned_process = pattern

        # Extract surface treatments
        surface_treatment_patterns = [
            "电镀", "镀锌", "镀铬", "镀镍", "化学镀",
            "阳极氧化", "硬质阳极", "本色阳极",
            "喷涂", "粉末涂装", "电泳", "达克罗",
            "发黑", "磷化", "钝化", "抛光", "喷砂",
        ]
        for pattern in surface_treatment_patterns:
            if pattern in content:
                ctx.surface_treatments.append(pattern)
                ctx.last_mentioned_process = pattern

        # Extract GD&T characteristics
        gdt_patterns = [
            "平面度", "直线度", "圆度", "圆柱度",
            "垂直度", "平行度", "倾斜度", "位置度",
            "同轴度", "对称度", "圆跳动", "全跳动",
            "轮廓度", "形位公差", "GD&T", "几何公差",
            "基准", "MMC", "LMC", "公差带",
        ]
        for pattern in gdt_patterns:
            if pattern in content:
                ctx.gdt_characteristics.append(pattern)

        # Update topic based on content keywords
        topic_keywords = {
            "材料": "material_selection",
            "强度": "material_properties",
            "公差": "tolerance",
            "配合": "fits",
            "螺纹": "threads",
            "轴承": "bearings",
            "加工": "machining",
            "切削": "cutting",
            "焊接": "welding",
            "热处理": "heat_treatment",
            "表面处理": "surface_treatment",
            "电镀": "electroplating",
            "阳极氧化": "anodizing",
            "GD&T": "gdt",
            "形位公差": "gdt",
            "几何公差": "gdt",
        }

        for keyword, topic in topic_keywords.items():
            if keyword in content:
                ctx.current_topic = topic
                if topic not in ctx.topic_history:
                    ctx.topic_history.append(topic)
                break

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history."""
        conv = self._conversations.get(conversation_id)
        if conv:
            conv.messages.clear()
            conv.context = ConversationContext()
            return True
        return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation entirely."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations with summary info."""
        return [
            {
                "id": conv.id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "current_topic": conv.context.current_topic,
            }
            for conv in self._conversations.values()
        ]


# Singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager singleton."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
