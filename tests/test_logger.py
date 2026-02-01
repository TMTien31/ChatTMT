import pytest
import logging
from pathlib import Path
from app.utils.logger import setup_logger, get_logger, clear_loggers
from app.utils.config import reload_config


class TestLoggerSetup:
    
    def test_setup_logger_default(self, monkeypatch, tmp_path):
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LOG_FILE", str(log_file))
        reload_config()
        
        logger = setup_logger("test_module")
        
        assert logger.name == "test_module"
        assert len(logger.handlers) > 0
    
    def test_setup_logger_creates_directory(self, monkeypatch, tmp_path):
        log_file = tmp_path / "logs" / "subdir" / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        
        logger = setup_logger("test", log_file=str(log_file))
        logger.info("Test")
        
        assert log_file.exists()


class TestLogLevels:
    """Test different log levels."""
    
    def test_debug_level(self, monkeypatch, tmp_path):
        """Test DEBUG level logs everything."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", level="DEBUG", log_file=str(log_file), log_to_console=False)
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        
        content = log_file.read_text()
        assert "Debug" in content
        assert "Info" in content
        assert "Warning" in content
    
    def test_info_level(self, monkeypatch, tmp_path):
        """Test INFO level filters DEBUG."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", level="INFO", log_file=str(log_file), log_to_console=False)
        logger.debug("Debug")
        logger.info("Info")
        
        content = log_file.read_text()
        assert "Debug" not in content
        assert "Info" in content
    
    def test_warning_level(self, monkeypatch, tmp_path):
        """Test WARNING level filters INFO."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", level="WARNING", log_file=str(log_file), log_to_console=False)
        logger.info("Info")
        logger.warning("Warning")
        
        content = log_file.read_text()
        assert "Info" not in content
        assert "Warning" in content


class TestFileOutput:
    """Test file logging."""
    
    def test_logs_to_file(self, monkeypatch, tmp_path):
        """Test logs written to file."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", log_file=str(log_file), log_to_console=False)
        logger.info("Test message")
        
        assert log_file.exists()
        assert "Test message" in log_file.read_text()
    
    def test_utf8_encoding(self, monkeypatch, tmp_path):
        """Test UTF-8 support."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", log_file=str(log_file), log_to_console=False)
        logger.info("Tiếng Việt: Xin chào")
        
        content = log_file.read_text(encoding='utf-8')
        assert "Tiếng Việt: Xin chào" in content


class TestConsoleOutput:
    """Test console logging."""
    
    def test_console_disabled(self, monkeypatch, tmp_path):
        """Test console can be disabled."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", log_file=str(log_file), log_to_console=False)
        
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" not in handler_types
    
    def test_console_enabled(self, monkeypatch, tmp_path):
        """Test console can be enabled."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = setup_logger("test", log_file=str(log_file), log_to_console=True)
        
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types


class TestGetLogger:
    """Test get_logger function."""
    
    def test_get_logger(self, monkeypatch):
        """Test get_logger returns logger."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_caches(self, monkeypatch):
        """Test get_logger caches loggers."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger1 = get_logger("test")
        logger2 = get_logger("test")
        
        assert logger1 is logger2
    
    def test_different_loggers(self, monkeypatch):
        """Test different names get different loggers."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reload_config()
        clear_loggers()
        
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1 is not logger2


class TestIntegration:
    """Test integration with config."""
    
    def test_uses_config(self, monkeypatch, tmp_path):
        """Test logger uses config values."""
        log_file = tmp_path / "app.log"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LOG_FILE", str(log_file))
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        reload_config()
        clear_loggers()
        
        logger = get_logger("test")
        logger.debug("Debug message")
        
        assert log_file.exists()
        assert "Debug message" in log_file.read_text()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
