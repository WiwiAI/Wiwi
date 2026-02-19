"""Event bus for inter-module communication."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import logging

from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message


# Type alias for event handlers
EventHandler = Callable[[Message], Awaitable[Optional[Message]]]


@dataclass
class Subscription:
    """Subscription to events on the bus."""

    subscriber: str                              # Module name
    handler: EventHandler                        # Async handler function
    port_filter: Optional[Set[PortType]] = None  # Filter by port types
    source_filter: Optional[Set[str]] = None     # Filter by source modules
    priority: int = 0                            # Higher = processed first


class EventBus:
    """
    Central event bus for asynchronous communication between modules.

    Implements Pub/Sub pattern with support for:
    - Asynchronous message processing
    - Port and source filtering
    - Handler priorities
    - Broadcast and targeted messages
    - Middleware for message transformation

    Usage:
        bus = EventBus()
        await bus.start()

        # Subscribe to events
        bus.subscribe(
            "my_module",
            my_handler,
            ports={PortType.TEXT_IN}
        )

        # Publish a message
        await bus.publish(message)

        await bus.stop()
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum number of messages in queue
        """
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._port_subscriptions: Dict[PortType, List[Subscription]] = defaultdict(list)
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("wiwi.event_bus")
        self._middleware: List[Callable[[Message], Awaitable[Message]]] = []
        self._message_count = 0

    @property
    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Current number of messages in queue."""
        return self._message_queue.qsize()

    @property
    def message_count(self) -> int:
        """Total number of processed messages."""
        return self._message_count

    async def start(self) -> None:
        """Start the event bus message processing."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_messages())
        self._logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the event bus and wait for queue to drain."""
        if not self._running:
            return

        self._running = False

        # Wait for remaining messages
        if not self._message_queue.empty():
            self._logger.info(
                f"Waiting for {self._message_queue.qsize()} messages to process"
            )
            await self._message_queue.join()

        # Cancel the processing task
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Event bus stopped")

    def subscribe(
        self,
        subscriber: str,
        handler: EventHandler,
        ports: Optional[Set[PortType]] = None,
        sources: Optional[Set[str]] = None,
        priority: int = 0
    ) -> None:
        """
        Subscribe to events.

        Args:
            subscriber: Name of the subscribing module
            handler: Async function to handle messages
            ports: Set of port types to listen to (None = all)
            sources: Set of source modules to listen to (None = all)
            priority: Handler priority (higher = processed first)
        """
        subscription = Subscription(
            subscriber=subscriber,
            handler=handler,
            port_filter=ports,
            source_filter=sources,
            priority=priority
        )

        # Index by subscriber
        self._subscriptions[subscriber].append(subscription)

        # Index by ports for fast lookup
        target_ports = ports if ports else set(PortType)
        for port in target_ports:
            self._port_subscriptions[port].append(subscription)
            # Keep sorted by priority (descending)
            self._port_subscriptions[port].sort(key=lambda s: -s.priority)

        self._logger.debug(
            f"Subscription added: {subscriber} -> ports={ports}, sources={sources}"
        )

    def unsubscribe(self, subscriber: str) -> None:
        """
        Remove all subscriptions for a module.

        Args:
            subscriber: Name of the module to unsubscribe
        """
        if subscriber not in self._subscriptions:
            return

        # Remove from subscriptions dict
        del self._subscriptions[subscriber]

        # Remove from port subscriptions
        for port in PortType:
            self._port_subscriptions[port] = [
                s for s in self._port_subscriptions[port]
                if s.subscriber != subscriber
            ]

        self._logger.debug(f"Subscription removed: {subscriber}")

    async def publish(self, message: Message) -> None:
        """
        Publish a message to the bus.

        Args:
            message: Message to publish

        Raises:
            asyncio.QueueFull: If queue is full
        """
        # Apply middleware
        for middleware in self._middleware:
            message = await middleware(message)

        await self._message_queue.put(message)
        self._logger.debug(
            f"Message published: {message.source} -> {message.target or 'broadcast'} "
            f"[{message.port.name}]"
        )

    async def publish_direct(self, message: Message) -> None:
        """
        Publish a message and dispatch immediately (bypass queue).

        Use this for real-time streaming where latency matters.
        Does not wait for handlers to complete.

        Args:
            message: Message to publish
        """
        # Apply middleware
        for middleware in self._middleware:
            message = await middleware(message)

        self._logger.debug(
            f"Message published (direct): {message.source} -> {message.target or 'broadcast'} "
            f"[{message.port.name}]"
        )

        # Dispatch immediately without waiting
        asyncio.create_task(self._dispatch_message(message))
        self._message_count += 1

    async def publish_and_wait(
        self,
        message: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """
        Publish a message and wait for a response.

        Args:
            message: Message to publish
            timeout: Timeout in seconds

        Returns:
            Response message or None if timeout
        """
        response_event = asyncio.Event()
        response_holder: List[Optional[Message]] = [None]

        async def response_handler(resp: Message) -> None:
            if resp.correlation_id == message.correlation_id:
                response_holder[0] = resp
                response_event.set()

        # Subscribe for response
        temp_subscriber = f"_temp_{message.correlation_id}"
        self.subscribe(
            temp_subscriber,
            response_handler,
            sources={message.target} if message.target else None
        )

        try:
            await self.publish(message)
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            return response_holder[0]
        except asyncio.TimeoutError:
            self._logger.warning(
                f"Timeout waiting for response: {message.correlation_id}"
            )
            return None
        finally:
            self.unsubscribe(temp_subscriber)

    def add_middleware(
        self,
        middleware: Callable[[Message], Awaitable[Message]]
    ) -> None:
        """
        Add middleware for message transformation.

        Middleware is called in order for each message before delivery.

        Args:
            middleware: Async function that transforms messages
        """
        self._middleware.append(middleware)

    async def _process_messages(self) -> None:
        """Background task for processing message queue."""
        while self._running:
            try:
                # Wait for message with timeout to allow checking _running flag
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._dispatch_message(message)
                self._message_count += 1
            except Exception as e:
                self._logger.error(f"Error dispatching message: {e}", exc_info=True)
            finally:
                self._message_queue.task_done()

    async def _dispatch_message(self, message: Message) -> None:
        """Dispatch message to all matching subscribers."""
        subscriptions = self._port_subscriptions.get(message.port, [])

        tasks = []
        for sub in subscriptions:
            # Check source filter
            if sub.source_filter and message.source not in sub.source_filter:
                continue

            # Check target filter (if message has specific target)
            if message.target and sub.subscriber != message.target:
                continue

            tasks.append(self._safe_call_handler(sub, message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.error(
                        f"Handler error in {subscriptions[i].subscriber}: {result}"
                    )
                elif isinstance(result, Message):
                    # Handler returned a response message - publish it
                    await self.publish(result)

    async def _safe_call_handler(
        self,
        subscription: Subscription,
        message: Message
    ) -> Optional[Message]:
        """Safely call a handler with error handling."""
        try:
            return await subscription.handler(message)
        except Exception as e:
            self._logger.error(
                f"Handler error in {subscription.subscriber}: {e}",
                exc_info=True
            )
            return None

    def get_subscribers(self, port: Optional[PortType] = None) -> List[str]:
        """
        Get list of subscriber names.

        Args:
            port: Optional port to filter by

        Returns:
            List of subscriber names
        """
        if port:
            return [s.subscriber for s in self._port_subscriptions.get(port, [])]
        return list(self._subscriptions.keys())
