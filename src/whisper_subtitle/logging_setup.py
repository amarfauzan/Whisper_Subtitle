import logging


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence third-party debug spam even in verbose mode: HTTP client
    # internals, numba JIT bytecode dumps, and onnx2torch converter
    # registration.
    for noisy in (
        "httpcore",
        "httpcore2",
        "httpx",
        "httpx2",
        "openai",
        "numba",
        "onnx2torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)