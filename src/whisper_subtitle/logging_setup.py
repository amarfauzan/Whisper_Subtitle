import logging

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence HTTP client noise even in verbose mode.
    for noisy in ("httpcore", "httpcore2", "httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)