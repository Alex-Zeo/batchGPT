from logs import logger, setup_logger

def test_logger_creates_files(tmp_path):
    setup_logger(tmp_path.as_posix())
    logger.info("info")
    logger.warning("warn")
    logger.error("err")
    assert (tmp_path / "info.log").exists()
    assert (tmp_path / "warning.log").exists()
    assert (tmp_path / "error.log").exists()
