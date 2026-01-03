# app/shared/events.py
"""
Pure Python implementation of the Event Bus / Signal-Slot pattern.

This module provides the `NativeEventBus` and `NativeSignal` classes to replace
Qt's signals when running in Headless/CLI mode. This removes the hard dependency
on PySide6 for non-GUI environments.
"""

import logging
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger("PixelHand.events")


class SignalProtocol(Protocol):
    """Protocol defining the interface for both Qt signals and Native signals."""

    def connect(self, callback: Callable) -> None: ...
    def emit(self, *args: Any) -> None: ...


class NativeSignal:
    """
    A pure Python implementation of a Signal (Observer pattern).

    It mimics the API of PySide6.QtCore.Signal so that business logic
    doesn't need to know if it's running in a GUI or CLI environment.
    """

    def __init__(self, *types: type):
        """
        Args:
            *types: Variable length argument list of types.
                    Kept for API compatibility with Qt Signal definitions,
                    but not strictly enforced in this lightweight implementation.
        """
        self._callbacks: list[Callable] = []

    def connect(self, callback: Callable):
        """Registers a callback function to be called when the signal is emitted."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback: Callable | None = None):
        """Unregisters a callback function."""
        if callback is None:
            self._callbacks.clear()
        elif callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, *args):
        """
        Triggers the signal, calling all registered callbacks synchronously.
        Exceptions in listeners are caught and logged to prevent crashing the emitter.
        """
        for callback in self._callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in signal handler '{callback.__name__}': {e}", exc_info=True)


class EventBusProtocol(Protocol):
    """
    Defines the contract for the Application Event Bus.
    This ensures both QtSignalBus and NativeEventBus expose the same signals.
    """

    # Scan Lifecycle
    scan_requested: SignalProtocol
    scan_started: SignalProtocol
    scan_finished: SignalProtocol
    scan_error: SignalProtocol
    scan_cancellation_requested: SignalProtocol

    # File Operations
    file_operation_started: SignalProtocol
    file_operation_finished: SignalProtocol

    # UI / System State
    lock_ui: SignalProtocol
    unlock_ui: SignalProtocol
    log_message: SignalProtocol
    show_error_dialog: SignalProtocol
    status_message_updated: SignalProtocol


class NativeEventBus:
    """
    The concrete Event Bus implementation for Headless/CLI mode.
    Uses NativeSignal instead of QObject/Signal.
    """

    def __init__(self):
        # Scan Lifecycle
        self.scan_requested = NativeSignal(object)  # payload: ScanConfig
        self.scan_started = NativeSignal()
        # payload: results_dict, num_found, mode_enum, duration_sec, skipped_files
        self.scan_finished = NativeSignal(object, int, object, float, list)
        self.scan_error = NativeSignal(str)
        self.scan_cancellation_requested = NativeSignal()

        # File Operations
        self.file_operation_started = NativeSignal(str)  # operation_name
        self.file_operation_finished = NativeSignal(list)  # affected_paths

        # UI / System State
        self.lock_ui = NativeSignal(str)  # reason
        self.unlock_ui = NativeSignal()
        self.log_message = NativeSignal(str, str)  # message, level
        self.show_error_dialog = NativeSignal(str, str)  # title, message
        self.status_message_updated = NativeSignal(str, int)  # message, timeout
