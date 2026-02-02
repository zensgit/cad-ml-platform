"""Message Queue System for CAD ML Platform.

Provides:
- Kafka producer/consumer
- RabbitMQ integration
- Event-driven architecture support
- Dead letter queue handling
"""

from src.core.messaging.producer import (
    MessageProducer,
    KafkaProducer,
    RabbitMQProducer,
    get_producer,
)
from src.core.messaging.consumer import (
    MessageConsumer,
    KafkaConsumer,
    RabbitMQConsumer,
    message_handler,
)
from src.core.messaging.events import (
    Event,
    EventType,
    EventBus,
    get_event_bus,
)

__all__ = [
    # Producer
    "MessageProducer",
    "KafkaProducer",
    "RabbitMQProducer",
    "get_producer",
    # Consumer
    "MessageConsumer",
    "KafkaConsumer",
    "RabbitMQConsumer",
    "message_handler",
    # Events
    "Event",
    "EventType",
    "EventBus",
    "get_event_bus",
]
