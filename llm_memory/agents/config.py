"""
Configuration for Multi-Agent System.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: Literal["ollama", "openai", "anthropic", "google", "azure"] = "ollama"
    model_name: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    
    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }
        
        env_var = env_vars.get(self.provider)
        if env_var:
            return os.getenv(env_var)
        
        return None


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system."""
    
    # Default model for all agents
    default_model: ModelConfig = field(default_factory=ModelConfig)
    
    # Override models for specific agents
    agent_models: dict[str, ModelConfig] = field(default_factory=dict)
    
    # Memory settings
    enable_memory: bool = True
    memory_persist_path: str = "./agent_memory"
    
    # Execution settings
    max_iterations: int = 10
    enable_human_in_loop: bool = False
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"
    
    def get_model_for_agent(self, agent_name: str) -> ModelConfig:
        """Get model config for a specific agent."""
        return self.agent_models.get(agent_name, self.default_model)


# Default configuration using Ollama
DEFAULT_CONFIG = AgentConfig(
    default_model=ModelConfig(
        provider="ollama",
        model_name="llama3.2",
        temperature=0.7,
    ),
    enable_memory=True,
    max_iterations=10,
)


# Configuration for using Gemma (larger model)
GEMMA_CONFIG = AgentConfig(
    default_model=ModelConfig(
        provider="ollama",
        model_name="gemma3:27b",
        temperature=0.7,
    ),
    enable_memory=True,
    max_iterations=10,
)


# Configuration for using Qwen
QWEN_CONFIG = AgentConfig(
    default_model=ModelConfig(
        provider="ollama",
        model_name="qwen2.5:7b",
        temperature=0.7,
    ),
    enable_memory=True,
    max_iterations=10,
)
