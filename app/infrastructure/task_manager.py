# app/infrastructure/task_manager.py
"""
Unified Task Manager for concurrency handling.
"""

import logging
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
            headless: If True, disables QThreadPool and uses standard Python threads.
            max_workers: Number of workers for the CPU-bound executor.
        """
        self.headless = headless

        # 1. GUI Thread Pool (Fire-and-forget, Signal support)
        self._qt_pool: QThreadPool | None = None
        if not headless:
            self._qt_pool = QThreadPool.globalInstance()
            # Reserve one thread for the main GUI loop interactions if needed
            self._qt_pool.setMaxThreadCount(max(2, max_workers + 2))

        # 2. Standard Thread Pool (For I/O-bound tasks needing Futures)
        self._std_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PixelHand_IO")

        # 3. Process Pool (For CPU-bound tasks to bypass GIL)
        # Used for image preprocessing/hashing
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
        Starts a fire-and-forget background task. Ideal for File I/O or UI updates.

        Args:
            func: The function to run.
            on_finish: Callback receiving the result (optional).
            on_error: Callback receiving an exception (optional).
            *args, **kwargs: Arguments passed to func.
        """
        if self.headless:
            # CLI Mode: Use standard executor and handle callbacks synchronously after completion
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
        Used for heavy calculations (Preprocessing, Hashing) to bypass the GIL.
        """
        return self._cpu_pool.submit(func, *args, **kwargs)

    def shutdown(self):
        """
        Cleanly shuts down all thread/process pools.
        """
        logger.info("Stopping TaskManager...")

        # Shutdown standard pool
        self._std_pool.shutdown(wait=True)
        self._cpu_pool.shutdown(wait=True)

        # QThreadPool (Global Instance) usually cleans itself up,
        # but we can wait if we want to ensure tasks finish.
        if self._qt_pool:
            self._qt_pool.waitForDone()
