"""CLI interface module for Wiwi4.0."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message

# Try to import rich for colored output
try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CLIModule(BaseModule):
    """
    Command-line interface module.

    Provides interactive text-based interface for user interaction.

    Features:
    - Colored output (with rich library)
    - Command history
    - Special commands (/help, /clear, /quit, etc.)
    - Streaming output support

    Config options:
        prompt: Input prompt string (default: ">>> ")
        assistant_prefix: Prefix for assistant responses (default: "Wiwi: ")
        colors: Enable colored output (default: true)
        history_file: Path to history file (default: "~/.wiwi_history")
    """

    COMMANDS = {
        "/help": "Show help message",
        "/clear": "Clear conversation history",
        "/quit": "Exit the assistant",
        "/exit": "Exit the assistant",
        "/modules": "List loaded modules",
        "/reload": "Reload a module",
        "/status": "Show system status",
        "/voice": "Toggle voice mode (speak to chat)",
        "/voice on": "Enable voice mode",
        "/voice off": "Disable voice mode",
        "/speak": "Toggle TTS (text-to-speech)",
        "/speak on": "Enable TTS",
        "/speak off": "Disable TTS",
        "/devices": "List available audio input devices",
        "/speakers": "List available audio output devices",
    }

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="cli_interface",
            version="1.0.0",
            description="Command-line interface for user interaction",
            author="Wiwi Team",
            category="interface",
            input_ports={PortType.TEXT_OUT, PortType.AUDIO_OUT, PortType.SYSTEM_OUT},
            output_ports={PortType.TEXT_IN, PortType.COMMAND_IN},
            dependencies=set(),
            optional_dependencies={"memory"},
            config_schema={
                "prompt": {"type": "string", "default": ">>> "},
                "assistant_prefix": {"type": "string", "default": "Wiwi: "},
                "colors": {"type": "boolean", "default": True},
                "history_file": {"type": "string", "default": "~/.wiwi_history"}
            }
        )

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)

        self._prompt = config.get("prompt", ">>> ")
        self._assistant_prefix = config.get("assistant_prefix", "Wiwi: ")
        self._use_colors = config.get("colors", True) and RICH_AVAILABLE
        self._history_file = Path(config.get("history_file", "~/.wiwi_history")).expanduser()

        self._console = Console() if RICH_AVAILABLE else None
        self._running = False

    async def initialize(self) -> None:
        """Initialize the CLI module."""
        # Try to load history
        if self._history_file.exists():
            self._logger.debug(f"History file found: {self._history_file}")

        self._logger.info("CLI interface initialized")

    async def start(self) -> None:
        """Start the CLI module."""
        self._state = ModuleState.RUNNING
        self._running = True

    async def stop(self) -> None:
        """Stop the CLI module."""
        self._running = False
        self._state = ModuleState.STOPPED

    async def handle_input(self, message: Message) -> Optional[Message]:
        """
        Handle incoming messages (responses from other modules).

        TEXT_OUT: Display LLM response (only if not from voice mode - voice prints during streaming)
        SYSTEM_OUT: Display system message
        """
        if message.port == PortType.TEXT_OUT:
            # LLM response - only display if NOT from voice mode
            # Voice mode already prints during streaming in LLM module
            if message.source == "llm_brain":
                # Skip if this was streamed (voice mode prints in real-time)
                if not message.metadata.get("streamed"):
                    await self._display_llm_response(message.payload)

        elif message.port == PortType.SYSTEM_OUT:
            await self._display_system_message(message.payload)

        return None

    async def _display_llm_response(self, text: str) -> None:
        """Display LLM response (for non-voice mode EventBus pipeline)."""
        if self._use_colors and self._console:
            self._console.print(f"[bold green]{self._assistant_prefix}[/]{text}")
        else:
            print(f"{self._assistant_prefix}{text}")

    async def run_interactive(self) -> None:
        """
        Run the interactive CLI loop.

        This is the main entry point for user interaction.
        """
        self._print_welcome()

        while self._running:
            try:
                # Get user input
                user_input = await self._get_input()

                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Send to processing pipeline
                await self._process_user_input(user_input)

            except KeyboardInterrupt:
                print()
                # Check if voice mode is active, disable it first
                stt = self.get_module("stt")
                if stt and stt.is_voice_mode_active:
                    await stt.disable_voice_mode()
                    self._print_info("Voice mode disabled. Press Ctrl+C again to exit.")
                else:
                    self._print_info("Press Ctrl+C again or type /quit to exit")
            except EOFError:
                break
            except Exception as e:
                self._logger.error(f"Error in interactive loop: {e}")
                self._print_error(f"Error: {e}")

        self._print_info("Goodbye!")

    async def _get_input(self) -> Optional[str]:
        """Get input from user."""
        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            if self._use_colors and self._console:
                # Use rich prompt
                return await loop.run_in_executor(
                    None,
                    lambda: self._console.input(f"[bold cyan]{self._prompt}[/]")
                )
            else:
                return await loop.run_in_executor(
                    None,
                    lambda: input(self._prompt)
                )
        except EOFError:
            return None

    async def _process_user_input(self, text: str) -> None:
        """Process user input with streaming output and realtime TTS."""
        llm = self.get_module("llm_brain")
        if not llm:
            self._print_error("LLM module not available")
            return

        # Get context from memory if available
        memory = self.get_module("memory")
        context = memory.get_context() if memory else []

        # Check if TTS is enabled for realtime streaming
        tts = self.get_module("tts")
        tts_enabled = tts and tts.is_tts_enabled

        # Print assistant prefix
        if self._use_colors and self._console:
            self._console.print(f"[bold green]{self._assistant_prefix}[/]", end="")
        else:
            print(f"{self._assistant_prefix}", end="", flush=True)

        # Stream response - check TTS streaming mode
        full_response = ""
        sentence_buffer = ""
        sentence_endings = ".!?"  # Only split on sentence-ending punctuation

        # Check if sentence-based streaming is enabled (for lower latency)
        # If false, send entire response to TTS at once (better voice quality)
        tts_config = tts._config if tts else {}
        sentence_streaming = tts_config.get("sentence_streaming", True)

        try:
            async for token in llm.generate_stream(text, context):
                print(token, end="", flush=True)
                full_response += token
                sentence_buffer += token

                # Check if we have a complete sentence to send to TTS
                # Only if sentence_streaming is enabled
                if tts_enabled and self._event_bus and sentence_streaming:
                    # Look for sentence endings
                    for i, char in enumerate(sentence_buffer):
                        if char in sentence_endings:
                            # Check if followed by space or end of buffer
                            if i == len(sentence_buffer) - 1 or sentence_buffer[i + 1] in ' \n':
                                # Extract the sentence
                                sentence = sentence_buffer[:i + 1].strip()
                                sentence_buffer = sentence_buffer[i + 1:].lstrip()

                                if sentence and len(sentence) > 2:  # Skip very short fragments
                                    # Send sentence to TTS immediately (direct dispatch)
                                    tts_message = Message(
                                        source="cli_interface",
                                        port=PortType.AUDIO_OUT,
                                        payload=sentence,
                                        target="tts",
                                        metadata={"streaming": True},
                                    )
                                    await self._event_bus.publish_direct(tts_message)
                                break

            print()  # Newline after response

            # Send remaining/full text to TTS
            if tts_enabled and self._event_bus:
                text_to_send = sentence_buffer.strip() if sentence_streaming else full_response.strip()
                if text_to_send:
                    tts_message = Message(
                        source="cli_interface",
                        port=PortType.AUDIO_OUT,
                        payload=text_to_send,
                        target="tts",
                        metadata={"streaming": True, "final": True},
                    )
                    await self._event_bus.publish_direct(tts_message)

        except Exception as e:
            self._logger.error(f"Generation error: {e}")
            self._print_error(f"Error: {e}")
            return

        # Save to memory if available
        if memory:
            memory.add_turn("user", text)
            memory.add_turn("assistant", full_response)

    async def _handle_command(self, command: str) -> bool:
        """
        Handle CLI commands.

        Returns:
            True to continue, False to exit
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            return False

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/clear":
            memory = self.get_module("memory")
            if memory:
                memory.clear_history()
                self._print_info("Conversation history cleared")
            else:
                self._print_error("Memory module not loaded")

        elif cmd == "/modules":
            await self._show_modules()

        elif cmd == "/status":
            await self._show_status()

        elif cmd == "/reload":
            if args:
                await self._reload_module(args)
            else:
                self._print_error("Usage: /reload <module_name>")

        elif cmd == "/voice":
            await self._handle_voice_command(args)

        elif cmd == "/speak":
            await self._handle_speak_command(args)

        elif cmd == "/devices":
            await self._show_audio_devices()

        elif cmd == "/speakers":
            await self._show_audio_output_devices()

        else:
            self._print_error(f"Unknown command: {cmd}")
            self._print_info("Use /help to see available commands")

        return True

    async def _handle_voice_command(self, args: str) -> None:
        """Handle /voice command."""
        stt = self.get_module("stt")
        if not stt:
            self._print_error("STT module not loaded. Add 'stt' to enabled_modules in config.")
            return

        args = args.lower().strip()

        if args == "on":
            await stt.enable_voice_mode()
            self._print_info("Voice mode enabled. Speak now...")
            # Also send command to TTS for auto-enable
            await self._send_command("voice_on")
        elif args == "off":
            await stt.disable_voice_mode()
            self._print_info("Voice mode disabled")
            # Also send command to TTS for auto-disable
            await self._send_command("voice_off")
        else:
            # Toggle
            if stt.is_voice_mode_active:
                await stt.disable_voice_mode()
                self._print_info("Voice mode disabled")
                await self._send_command("voice_off")
            else:
                await stt.enable_voice_mode()
                self._print_info("Voice mode enabled. Speak now...")
                await self._send_command("voice_on")

    async def _handle_speak_command(self, args: str) -> None:
        """Handle /speak command."""
        tts = self.get_module("tts")
        if not tts:
            self._print_error("TTS module not loaded. Add 'tts' to enabled_modules in config.")
            return

        args = args.lower().strip()

        if args == "on":
            await tts.enable_tts()
            self._print_info("TTS enabled")
        elif args == "off":
            await tts.disable_tts()
            self._print_info("TTS disabled")
        elif args == "stop":
            tts.stop_speaking()
            self._print_info("TTS playback stopped")
        else:
            # Toggle
            if tts.is_tts_enabled:
                await tts.disable_tts()
                self._print_info("TTS disabled")
            else:
                await tts.enable_tts()
                self._print_info("TTS enabled")

    async def _send_command(self, command: str) -> None:
        """Send a command to all modules via event bus."""
        if self._event_bus:
            message = Message(
                source=self.name,
                port=PortType.COMMAND_IN,
                payload=command,
                target=None,  # Broadcast
            )
            await self._event_bus.publish(message)

    async def _show_audio_devices(self) -> None:
        """Show available audio input devices."""
        try:
            from wiwi.modules.stt.audio_capture import AudioCapture
            devices = AudioCapture.list_devices()

            if not devices:
                self._print_error("No audio input devices found")
                return

            if self._use_colors and self._console:
                self._console.print("[bold]Audio input devices:[/]")
                for dev in devices:
                    default_marker = " [green](default)[/]" if dev.get('is_default') else ""
                    self._console.print(
                        f"  [{dev['index']}] [cyan]{dev['name']}[/]{default_marker}"
                    )
            else:
                print("Audio input devices:")
                for dev in devices:
                    default_marker = " (default)" if dev.get('is_default') else ""
                    print(f"  [{dev['index']}] {dev['name']}{default_marker}")
            print()
        except ImportError:
            self._print_error("STT module not available")

    async def _show_audio_output_devices(self) -> None:
        """Show available audio output devices."""
        try:
            from wiwi.modules.tts.audio_output import StreamingAudioPlayer
            devices = StreamingAudioPlayer.list_output_devices()

            if not devices:
                self._print_error("No audio output devices found")
                return

            if self._use_colors and self._console:
                self._console.print("[bold]Audio output devices:[/]")
                for dev in devices:
                    default_marker = " [green](default)[/]" if dev.get('is_default') else ""
                    self._console.print(
                        f"  [{dev['index']}] [cyan]{dev['name']}[/]{default_marker}"
                    )
            else:
                print("Audio output devices:")
                for dev in devices:
                    default_marker = " (default)" if dev.get('is_default') else ""
                    print(f"  [{dev['index']}] {dev['name']}{default_marker}")
            print()
        except ImportError:
            self._print_error("TTS module not available")

    async def _display_system_message(self, text: str) -> None:
        """Display system message."""
        self._print_info(text)

    def _print_welcome(self) -> None:
        """Print welcome message."""
        if self._use_colors and self._console:
            self._console.print(Panel(
                "[bold]Wiwi4.0[/bold] - Modular AI Assistant\n"
                "Type [cyan]/help[/] for available commands",
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("=" * 50)
            print("Wiwi4.0 - Modular AI Assistant")
            print("Type /help for available commands")
            print("=" * 50)
        print()

    def _print_help(self) -> None:
        """Print help message."""
        if self._use_colors and self._console:
            lines = ["[bold]Available commands:[/]\n"]
            for cmd, desc in self.COMMANDS.items():
                lines.append(f"  [cyan]{cmd}[/] - {desc}")
            self._console.print("\n".join(lines))
        else:
            print("Available commands:")
            for cmd, desc in self.COMMANDS.items():
                print(f"  {cmd} - {desc}")
        print()

    def _print_info(self, text: str) -> None:
        """Print info message."""
        if self._use_colors and self._console:
            self._console.print(f"[blue]ℹ {text}[/]")
        else:
            print(f"[INFO] {text}")

    def _print_error(self, text: str) -> None:
        """Print error message."""
        if self._use_colors and self._console:
            self._console.print(f"[red]✗ {text}[/]")
        else:
            print(f"[ERROR] {text}")

    async def _show_modules(self) -> None:
        """Show loaded modules."""
        modules = self._kernel.list_modules()

        if self._use_colors and self._console:
            self._console.print("[bold]Loaded modules:[/]")
            for name, info in modules.items():
                self._console.print(
                    f"  [cyan]{name}[/] v{info.version} - {info.description}"
                )
        else:
            print("Loaded modules:")
            for name, info in modules.items():
                print(f"  {name} v{info.version} - {info.description}")

        # Show available but not loaded
        available = set(self._kernel.list_available_modules())
        loaded = set(modules.keys())
        not_loaded = available - loaded

        if not_loaded:
            if self._use_colors and self._console:
                self._console.print("\n[bold]Available (not loaded):[/]")
                for name in not_loaded:
                    self._console.print(f"  [dim]{name}[/]")
            else:
                print("\nAvailable (not loaded):")
                for name in not_loaded:
                    print(f"  {name}")
        print()

    async def _show_status(self) -> None:
        """Show system status."""
        if self._use_colors and self._console:
            self._console.print("[bold]System Status:[/]")
            self._console.print(f"  Kernel: [green]Running[/]")
            self._console.print(f"  Event Bus: [green]Running[/] ({self._event_bus.message_count} messages processed)")
            self._console.print(f"  Modules: {len(self._kernel.list_modules())} loaded")
        else:
            print("System Status:")
            print(f"  Kernel: Running")
            print(f"  Event Bus: Running ({self._event_bus.message_count} messages)")
            print(f"  Modules: {len(self._kernel.list_modules())} loaded")

        # Memory status
        memory = self.get_module("memory")
        if memory:
            if self._use_colors and self._console:
                self._console.print(f"  Memory: {memory.history_length} messages in history")
            else:
                print(f"  Memory: {memory.history_length} messages in history")
        print()

    async def _reload_module(self, module_name: str) -> None:
        """Reload a module."""
        self._print_info(f"Reloading module: {module_name}")
        try:
            await self._kernel.reload_module(module_name)
            self._print_info(f"Module {module_name} reloaded successfully")
        except Exception as e:
            self._print_error(f"Failed to reload {module_name}: {e}")
