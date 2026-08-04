from pathlib import Path

from core.logger import get_logger


def test_logger_emits_info_messages():
    logger = get_logger("test_logger")
    logger.info("logger test message")

    log_file = Path("logs/test_logger.log")
    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")
    assert "logger test message" in content
