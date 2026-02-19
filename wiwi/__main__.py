"""Entry point for Wiwi4.0."""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

from wiwi.core.kernel import Kernel
from wiwi.utils.logging_setup import setup_logging
import uvloop

uvloop.install()

# Global shutdown flag for signal handlers
_shutdown_requested = False
_kernel_instance = None


async def main(config_path: Path = None) -> None:
    """
    Main entry point for Wiwi4.0.

    Args:
        config_path: Path to configuration file
    """
    global _shutdown_requested, _kernel_instance

    # Create kernel
    kernel = Kernel(config_path)
    _kernel_instance = kernel

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    def signal_handler():
        global _shutdown_requested
        if _shutdown_requested:
            # Second Ctrl+C - force exit
            print("\nForce exit...")
            sys.exit(1)
        _shutdown_requested = True
        print("\nShutting down gracefully... (Press Ctrl+C again to force)")
        shutdown_event.set()
        # Also stop CLI if running
        cli = kernel.get_module("cli_interface")
        if cli:
            cli._running = False

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())

    try:
        # Start kernel
        await kernel.start()

        # Get CLI module
        cli = kernel.get_module("cli_interface")

        if cli:
            # Run interactive mode
            await cli.run_interactive()
        else:
            # No CLI module - wait for shutdown signal
            print("No CLI module loaded. Press Ctrl+C to exit.")
            await shutdown_event.wait()

    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    finally:
        _shutdown_requested = True
        await kernel.stop()
        _kernel_instance = None


def cli_entry() -> None:
    """
    CLI entry point.

    This is called when running: python -m wiwi or wiwi command.
    """
    parser = argparse.ArgumentParser(
        description="Wiwi4.0 - Modular AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wiwi                      Start interactive mode
  wiwi -c config.yaml       Start with custom config
  wiwi --list-modules       List available modules
  wiwi --version            Show version

For more information, visit: https://github.com/wiwi/wiwi4
"""
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List available modules and exit"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file"
    )

    args = parser.parse_args()

    # Show version
    if args.version:
        from wiwi import __version__
        print(f"Wiwi v{__version__}")
        return

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

    # List modules
    if args.list_modules:
        asyncio.run(list_modules(args.config))
        return

    # Run main
    try:
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        pass


async def list_modules(config_path: Path = None) -> None:
    """List all available modules."""
    kernel = Kernel(config_path)
    await kernel.discover_modules()

    print("\nAvailable modules:")
    print("-" * 50)

    for name in sorted(kernel.list_available_modules()):
        info = kernel.registry.get_info(name)
        if info:
            print(f"  {name:<20} v{info.version}")
            print(f"    {info.description}")
            if info.dependencies:
                print(f"    Dependencies: {', '.join(info.dependencies)}")
            print()
        else:
            print(f"  {name}")


if __name__ == "__main__":
    cli_entry()
