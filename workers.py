"""Worker processes must not outlive the app (Quit, crash or closed window): they would keep PDFs locked."""
import multiprocessing
import os
import threading


def exit_with_parent() -> None:
    """ProcessPoolExecutor initializer: end this worker as soon as the app's process is gone."""
    parent = multiprocessing.parent_process()
    if parent is not None:
        def watch() -> None:
            parent.join()
            os._exit(1)
        threading.Thread(target=watch, daemon=True).start()
