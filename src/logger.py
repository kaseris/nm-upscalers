import logging
from pathlib import Path

class _PathFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            rel = Path(record.pathname).relative_to(Path.cwd())
        except ValueError:
            rel = Path(record.pathname)
        record.module_path = str(rel.with_suffix("")).replace("/", ".")
        return super().format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(
    _PathFormatter("%(asctime)s - [%(module_path)s] - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
