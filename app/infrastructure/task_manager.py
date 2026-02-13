# app/infrastructure/task_manager.py
"""
Unified Task Manager for concurrency handling.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

logger = logging.getLogger("PixelHand.task_manager")


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Must inherit QObject to use Signals.
    """

    finished = Signal(object)  # Emits the return value of the function
    error = Signal(Exception)  # Emits any exception raised


class RunnableWrapper(QRunnable):
    """
    Wraps a standard Python function into a QRunnable for QThreadPool.
    Handles exception capture and signal emission.
    """

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            logger.error(f"Task failed: {e}", exc_info=True)
            self.signals.error.emit(e)


class TaskManager:
    """
    Manages background threads and execution pools.
    """

    def __init__(self, headless: bool = False, max_workers: int = 4):
        """
        Args:
            headless: If True, disables QThreadPool usage logic (though QObject can still exist).
            max_workers: Number of workers for the CPU-bound executor.
        """
        self.headless = headless
        self._active_threads: list[threading.Thread] = []

        # 1. GUI Thread Pool (Fire-and-forget, Signal support)
        self._qt_pool: QThreadPool | None = None
        if not headless:
            self._qt_pool = QThreadPool.globalInstance()
            self._qt_pool.setMaxThreadCount(max(2, max_workers + 2))

        # 2. Standard Thread Pool (For I/O-bound tasks needing Futures)
        self._std_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PixelHand_IO")

        # 3. Process Pool (For CPU-bound tasks to bypass GIL)
        self._cpu_pool = ProcessPoolExecutor(max_workers=max_workers)

        logger.info(f"TaskManager initialized (Headless: {headless}, Workers: {max_workers})")

    def start_background_task(
        self,
        func: Callable,
        on_finish: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        *args,
        **kwargs,
    ):
        """
        Starts a fire-and-forget background task.
        """
        if self.headless:
            # CLI Mode: Use standard executor
            future = self._std_pool.submit(func, *args, **kwargs)

            def _callback_wrapper(fut: Future):
                try:
                    res = fut.result()
                    if on_finish:
                        on_finish(res)
                except Exception as e:
                    if on_error:
                        on_error(e)
                    else:
                        logger.error(f"Background task error: {e}")

            future.add_done_callback(_callback_wrapper)

        else:
            # GUI Mode: Use QRunnable with Signals
            runnable = RunnableWrapper(func, *args, **kwargs)
            if on_finish:
                runnable.signals.finished.connect(on_finish)
            if on_error:
                runnable.signals.error.connect(on_error)

            if self._qt_pool:
                self._qt_pool.start(runnable)

    def submit_cpu_bound(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submits a task to the ProcessPoolExecutor and returns a Future.
        """
        return self._cpu_pool.submit(func, *args, **kwargs)

    def start_thread(self, target: Callable, name: str, args: tuple = (), daemon: bool = True) -> threading.Thread:
        """
        Starts a standard python thread and tracks it.
        Used for long-running producer/consumer loops that don't fit into a thread pool.
        """
        t = threading.Thread(target=target, name=name, args=args, daemon=daemon)
        t.start()
        self._active_threads.append(t)
        # Cleanup finished threads from list
        self._active_threads = [t for t in self._active_threads if t.is_alive()]
        return t

    def shutdown(self):
        """
        Cleanly shuts down all thread/process pools.
        """
        logger.info("Stopping TaskManager...")

        self._std_pool.shutdown(wait=True)
        self._cpu_pool.shutdown(wait=True)

        if self._qt_pool:
            self._qt_pool.waitForDone()

        # Join explicit threads
        for t in self._active_threads:
            if t.is_alive():
                t.join(timeout=1.0)
