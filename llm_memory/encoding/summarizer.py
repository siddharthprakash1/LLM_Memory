"""
Memory summarization using LLMs.

Supports multiple providers:
- Ollama (local, default)
- OpenAI
- Anthropic
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from llm_memory.config import LLMConfig


# Prompt templates for different summarization tasks
SUMMARIZE_PROMPT = """Summarize the following content concisely while preserving key information:

Content:
{content}

Summary (be brief but capture essential details):"""

EXTRACT_FACTS_PROMPT = """Extract key facts from the following content. Return each fact on a new line.

Content:
{content}

Key facts:"""

EXTRACT_PREFERENCES_PROMPT = """Extract any user preferences or opinions from the following content. Return each preference on a new line.

Content:
{content}

Preferences:"""

CONSOLIDATE_EPISODES_PROMPT = """Analyze these related experiences and extract the general pattern or lesson learned:

Experiences:
{episodes}

What general insight or pattern emerges from these experiences?"""


class SummaryResult(BaseModel):
    """Result of a summarization operation."""

    original_length: int
    summary: str
    summary_length: int
    compression_ratio: float
    model: str


class BaseSummarizer(ABC):
    """Abstract base class for LLM summarizers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        pass

    async def summarize(self, content: str) -> str:
        """Summarize the given content."""
        prompt = SUMMARIZE_PROMPT.format(content=content)
        return await self.generate(prompt)

    async def summarize_with_metadata(self, content: str) -> SummaryResult:
        """Summarize with metadata about the operation."""
        summary = await self.summarize(content)
        return SummaryResult(
            original_length=len(content),
            summary=summary,
            summary_length=len(summary),
            compression_ratio=len(summary) / len(content) if content else 0,
            model=self.config.model,
        )

    async def extract_facts(self, content: str) -> list[str]:
        """Extract factual statements from content."""
        prompt = EXTRACT_FACTS_PROMPT.format(content=content)
        response = await self.generate(prompt)
        
        # Parse line-separated facts
        facts = [
            line.strip().lstrip("- •*")
            for line in response.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        return facts

    async def extract_preferences(self, content: str) -> list[str]:
        """Extract user preferences from content."""
        prompt = EXTRACT_PREFERENCES_PROMPT.format(content=content)
        response = await self.generate(prompt)

        # Parse line-separated preferences
        prefs = [
            line.strip().lstrip("- •*")
            for line in response.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        return prefs

    async def consolidate_episodes(self, episodes: list[str]) -> str:
        """Extract patterns from multiple episodes."""
        episodes_text = "\n\n---\n\n".join(episodes)
        prompt = CONSOLIDATE_EPISODES_PROMPT.format(episodes=episodes_text)
        return await self.generate(prompt)


class OllamaSummarizer(BaseSummarizer):
    """
    Ollama-based summarizer for local LLM operations.
    
    Uses Ollama's chat/generate API with models like:
    - llama3.2 (fast, good quality)
    - mistral (good balance)
    - phi3 (smaller, faster)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.ollama_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for LLM
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(self, prompt: str) -> str:
        """Generate text using Ollama."""
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["response"].strip()

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """Chat-style generation using Ollama."""
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"].strip()

    async def __aenter__(self) -> "OllamaSummarizer":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class OpenAISummarizer(BaseSummarizer):
    """
    OpenAI-based summarizer.
    
    Uses OpenAI's chat API with models like:
    - gpt-4o-mini (fast, cost-effective)
    - gpt-4o (best quality)
    - gpt-3.5-turbo (legacy, cheaper)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required for OpenAI summarization")
        return self._client

    async def generate(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content.strip()


class AnthropicSummarizer(BaseSummarizer):
    """
    Anthropic-based summarizer.
    
    Uses Anthropic's messages API with models like:
    - claude-3-5-sonnet (best quality)
    - claude-3-haiku (fast, cost-effective)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic()
            except ImportError:
                raise ImportError("anthropic package required for Anthropic summarization")
        return self._client

    async def generate(self, prompt: str) -> str:
        """Generate text using Anthropic."""
        client = self._get_client()

        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()


def create_summarizer(config: LLMConfig | None = None) -> BaseSummarizer:
    """
    Factory function to create the appropriate summarizer.
    
    Args:
        config: LLM configuration. Uses defaults if None.
        
    Returns:
        Configured summarizer instance.
    """
    if config is None:
        config = LLMConfig()

    providers = {
        "ollama": OllamaSummarizer,
        "openai": OpenAISummarizer,
        "anthropic": AnthropicSummarizer,
    }

    summarizer_class = providers.get(config.provider)
    if summarizer_class is None:
        raise ValueError(f"Unknown LLM provider: {config.provider}")

    return summarizer_class(config)


class MemorySummarizer:
    """
    High-level summarizer for memory operations.
    
    Provides memory-specific summarization methods.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._summarizer = create_summarizer(self.config)

    async def summarize_conversation(self, messages: list[dict[str, str]]) -> str:
        """
        Summarize a conversation into a concise description.
        
        Args:
            messages: List of messages with 'role' and 'content'.
            
        Returns:
            Summary of the conversation.
        """
        # Format conversation
        formatted = "\n".join(
            f"[{msg['role']}]: {msg['content']}"
            for msg in messages
        )

        prompt = f"""Summarize this conversation concisely, capturing:
1. Main topic/task
2. Key decisions or outcomes
3. Important information exchanged

Conversation:
{formatted}

Summary:"""

        return await self._summarizer.generate(prompt)

    async def extract_memory_content(
        self,
        content: str,
        extract_facts: bool = True,
        extract_preferences: bool = True,
    ) -> dict[str, Any]:
        """
        Extract structured information from content for memory storage.
        
        Returns dict with:
        - summary: Brief summary
        - facts: List of factual statements
        - preferences: List of preferences (if any)
        """
        result: dict[str, Any] = {}

        # Run extractions in parallel
        tasks = [self._summarizer.summarize(content)]
        
        if extract_facts:
            tasks.append(self._summarizer.extract_facts(content))
        if extract_preferences:
            tasks.append(self._summarizer.extract_preferences(content))

        results = await asyncio.gather(*tasks)

        result["summary"] = results[0]
        
        idx = 1
        if extract_facts:
            result["facts"] = results[idx]
            idx += 1
        if extract_preferences:
            result["preferences"] = results[idx]

        return result

    async def create_semantic_from_episodes(
        self,
        episode_descriptions: list[str],
    ) -> dict[str, str]:
        """
        Create semantic memory content from multiple episodes.
        
        Extracts patterns and generalizations from specific experiences.
        
        Returns dict with:
        - pattern: The generalized pattern
        - summary: Brief summary
        """
        pattern = await self._summarizer.consolidate_episodes(episode_descriptions)
        summary = await self._summarizer.summarize(pattern)

        return {
            "pattern": pattern,
            "summary": summary,
        }

    async def score_importance(self, content: str) -> dict[str, float]:
        """
        Use LLM to score content importance factors.
        
        Returns importance scores for:
        - emotional_salience
        - novelty
        - causal_significance
        """
        prompt = f"""Rate the following content on these scales (0.0 to 1.0):

1. Emotional salience: How emotionally significant is this? (frustration, excitement, satisfaction)
2. Novelty: How new or unique is this information?
3. Causal significance: How likely is this to affect future decisions or outcomes?

Content:
{content}

Respond with three numbers only, one per line (e.g., 0.7, 0.5, 0.8):"""

        response = await self._summarizer.generate(prompt)

        # Parse scores
        try:
            lines = response.strip().split("\n")
            scores = []
            for line in lines[:3]:
                # Extract number from line
                num_str = "".join(c for c in line if c.isdigit() or c == ".")
                if num_str:
                    scores.append(float(num_str))
            
            while len(scores) < 3:
                scores.append(0.5)  # Default

            return {
                "emotional_salience": min(1.0, max(0.0, scores[0])),
                "novelty": min(1.0, max(0.0, scores[1])),
                "causal_significance": min(1.0, max(0.0, scores[2])),
            }
        except (ValueError, IndexError):
            # Return defaults on parse error
            return {
                "emotional_salience": 0.5,
                "novelty": 0.5,
                "causal_significance": 0.5,
            }

    async def close(self) -> None:
        """Close summarizer resources."""
        if hasattr(self._summarizer, "close"):
            await self._summarizer.close()

    async def __aenter__(self) -> "MemorySummarizer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
