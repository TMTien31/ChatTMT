import pytest
import os
from app.utils.config import Config, get_config, reload_config


class TestConfigDefaults:
    
    def test_default_values(self):
        config = Config()
        
        assert config.OPENAI_MODEL == "gpt-4"
        assert config.OPENAI_TEMPERATURE == 0.2
        assert config.OPENAI_MAX_TOKENS == 2000
        
        assert config.TOKEN_THRESHOLD_RAW == 10000
        assert config.SUMMARY_TOKEN_THRESHOLD == 2000
        assert config.KEEP_RECENT_N == 16
        
        assert config.LIGHT_CONTEXT_SIZE == 8
        assert config.RECENT_CONTEXT_SIZE == 10
        assert config.MAX_CLARIFICATION_ROUNDS == 2
        
        # Session
        assert config.SESSION_DATA_DIR == "data/sessions"
        
        # Logging
        assert config.LOG_LEVEL == "INFO"
        assert config.LOG_FILE == "logs/app.log"
        assert config.LOG_TO_CONSOLE is True


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_with_valid_config(self, monkeypatch):
        """Test validation passes with valid config."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        
        # Should not raise
        config.validate()
    
    def test_validate_missing_api_key(self):
        """Test validation fails without API key."""
        config = Config()
        config.OPENAI_API_KEY = ""
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            config.validate()
    
    def test_validate_invalid_token_threshold(self, monkeypatch):
        """Test validation fails with invalid token threshold."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        config.TOKEN_THRESHOLD_RAW = 0
        
        with pytest.raises(ValueError, match="TOKEN_THRESHOLD_RAW must be positive"):
            config.validate()
    
    def test_validate_invalid_summary_threshold(self, monkeypatch):
        """Test validation fails with invalid summary threshold."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        config.SUMMARY_TOKEN_THRESHOLD = -100
        
        with pytest.raises(ValueError, match="SUMMARY_TOKEN_THRESHOLD must be positive"):
            config.validate()
    
    def test_validate_invalid_keep_recent_n(self, monkeypatch):
        """Test validation fails with invalid KEEP_RECENT_N."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        config.KEEP_RECENT_N = 0
        
        with pytest.raises(ValueError, match="KEEP_RECENT_N must be positive"):
            config.validate()
    
    def test_validate_invalid_clarification_rounds(self, monkeypatch):
        """Test validation fails with negative clarification rounds."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        config.MAX_CLARIFICATION_ROUNDS = -1
        
        with pytest.raises(ValueError, match="MAX_CLARIFICATION_ROUNDS must be non-negative"):
            config.validate()


class TestEnvironmentVariables:
    """Test loading from environment variables."""
    
    def test_load_from_env(self, monkeypatch):
        """Test configuration loads from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("TOKEN_THRESHOLD_RAW", "15000")
        monkeypatch.setenv("KEEP_RECENT_N", "20")
        
        config = reload_config()
        
        assert config.OPENAI_API_KEY == "sk-custom-key"
        assert config.OPENAI_MODEL == "gpt-3.5-turbo"
        assert config.TOKEN_THRESHOLD_RAW == 15000
        assert config.KEEP_RECENT_N == 20
    
    def test_temperature_as_float(self, monkeypatch):
        """Test temperature is parsed as float."""
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")
        config = reload_config()
        
        assert config.OPENAI_TEMPERATURE == 0.5
        assert isinstance(config.OPENAI_TEMPERATURE, float)
    
    def test_boolean_parsing(self, monkeypatch):
        """Test boolean values are parsed correctly."""
        monkeypatch.setenv("LOG_TO_CONSOLE", "false")
        config = reload_config()
        
        assert config.LOG_TO_CONSOLE is False
        
        monkeypatch.setenv("LOG_TO_CONSOLE", "True")
        config = reload_config()
        
        assert config.LOG_TO_CONSOLE is True


class TestSingleton:
    """Test singleton pattern."""
    
    def test_get_config_returns_same_instance(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reload_config_returns_new_instance(self):
        """Test reload_config creates new instance."""
        config1 = get_config()
        config2 = reload_config()
        
        # Should be new instance
        assert config1 is not config2


class TestConfigUsage:
    """Test practical configuration usage."""
    
    def test_access_config_values(self, monkeypatch):
        """Test accessing config values in typical usage."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = reload_config()
        
        # Simulate usage in modules
        threshold = config.TOKEN_THRESHOLD_RAW
        assert threshold == 10000
        
        model = config.OPENAI_MODEL
        assert model == "gpt-4"
        
        keep_n = config.KEEP_RECENT_N
        assert keep_n == 16


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
