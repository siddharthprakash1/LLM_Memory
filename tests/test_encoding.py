"""
Tests for encoding module (embedder and summarizer).

Note: Tests that require Ollama server are marked with @pytest.mark.ollama
and will be skipped if Ollama is not running.
"""

import pytest
import httpx

from llm_memory.config import EmbeddingConfig, LLMConfig
from llm_memory.encoding.embedder import (
    OllamaEmbedder,
    MemoryEmbedder,
    create_embedder,
    EmbeddingResult,
)
from llm_memory.encoding.summarizer import (
    OllamaSummarizer,
    MemorySummarizer,
    create_summarizer,
    SummaryResult,
)


def is_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip marker for tests requiring Ollama
ollama_required = pytest.mark.skipif(
    not is_ollama_running(),
    reason="Ollama server not running"
)


class TestEmbeddingConfig:
    """Tests for embedding configuration."""

    def test_default_config(self):
        """Test default embedding config uses Ollama."""
        config = EmbeddingConfig()
        assert config.provider == "ollama"
        assert config.model == "nomic-embed-text"
        assert config.dimensions == 768

    def test_custom_config(self):
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        )
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"


class TestCreateEmbedder:
    """Tests for embedder factory function."""

    def test_create_ollama_embedder(self):
        """Test creating Ollama embedder."""
        config = EmbeddingConfig(provider="ollama")
        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)

    def test_create_default_embedder(self):
        """Test creating embedder with default config."""
        embedder = create_embedder()
        assert isinstance(embedder, OllamaEmbedder)

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        config = EmbeddingConfig()
        config.provider = "invalid"  # type: ignore
        
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder(config)


@ollama_required
class TestOllamaEmbedder:
    """Tests for Ollama embedder (requires running Ollama server)."""

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Test embedding a single text."""
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
        )
        
        async with OllamaEmbedder(config) as embedder:
            embedding = await embedder.embed("Hello, world!")
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        config = EmbeddingConfig(provider="ollama")
        
        async with OllamaEmbedder(config) as embedder:
            texts = ["First text", "Second text", "Third text"]
            embeddings = await embedder.embed_batch(texts)
            
            assert len(embeddings) == 3
            assert all(len(e) > 0 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embed_with_metadata(self):
        """Test embedding with metadata."""
        config = EmbeddingConfig(provider="ollama")
        
        async with OllamaEmbedder(config) as embedder:
            result = await embedder.embed_with_metadata("Test content")
            
            assert isinstance(result, EmbeddingResult)
            assert result.text == "Test content"
            assert result.model == config.model
            assert result.dimensions > 0


@ollama_required
class TestMemoryEmbedder:
    """Tests for high-level MemoryEmbedder."""

    @pytest.mark.asyncio
    async def test_embed_memory_content(self):
        """Test embedding memory content."""
        async with MemoryEmbedder() as embedder:
            embedding = await embedder.embed_memory_content(
                content="User prefers dark mode in all applications.",
                summary="User preference: dark mode",
            )
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test embedding a query."""
        async with MemoryEmbedder() as embedder:
            embedding = await embedder.embed_query("What theme does the user prefer?")
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    @pytest.mark.asyncio
    async def test_embedding_cache(self):
        """Test that embeddings are cached."""
        async with MemoryEmbedder() as embedder:
            content = "This is test content"
            
            # First call
            embedding1 = await embedder.embed_memory_content(content)
            
            # Second call (should use cache)
            embedding2 = await embedder.embed_memory_content(content)
            
            assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        embedder = MemoryEmbedder()
        
        # Add something to cache
        await embedder.embed_memory_content("Test")
        assert len(embedder._cache) > 0
        
        # Clear cache
        embedder.clear_cache()
        assert len(embedder._cache) == 0
        
        await embedder.close()


class TestSummarizerConfig:
    """Tests for summarizer configuration."""

    def test_default_config(self):
        """Test default LLM config uses Ollama."""
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3.2"

    def test_custom_config(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.5,
        )
        assert config.provider == "openai"
        assert config.temperature == 0.5


class TestCreateSummarizer:
    """Tests for summarizer factory function."""

    def test_create_ollama_summarizer(self):
        """Test creating Ollama summarizer."""
        config = LLMConfig(provider="ollama")
        summarizer = create_summarizer(config)
        assert isinstance(summarizer, OllamaSummarizer)

    def test_create_default_summarizer(self):
        """Test creating summarizer with default config."""
        summarizer = create_summarizer()
        assert isinstance(summarizer, OllamaSummarizer)

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        config = LLMConfig()
        config.provider = "invalid"  # type: ignore
        
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_summarizer(config)


@ollama_required
class TestOllamaSummarizer:
    """Tests for Ollama summarizer (requires running Ollama server)."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test basic text generation."""
        config = LLMConfig(provider="ollama", max_tokens=100)
        
        async with OllamaSummarizer(config) as summarizer:
            response = await summarizer.generate("Say hello in one word.")
            
            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_summarize(self):
        """Test content summarization."""
        config = LLMConfig(provider="ollama", max_tokens=200)
        
        content = """
        The user has been working on a Python web application using FastAPI.
        They encountered several issues with authentication and spent time
        debugging JWT token validation. Eventually they found that the token
        expiration time was not being set correctly. After fixing the issue,
        all tests passed successfully.
        """
        
        async with OllamaSummarizer(config) as summarizer:
            summary = await summarizer.summarize(content)
            
            assert isinstance(summary, str)
            assert len(summary) < len(content)

    @pytest.mark.asyncio
    async def test_summarize_with_metadata(self):
        """Test summarization with metadata."""
        config = LLMConfig(provider="ollama", max_tokens=200)
        
        # Use longer content to ensure compression
        long_content = """
        This is a much longer piece of content that contains multiple sentences
        and paragraphs. It discusses various topics including programming,
        machine learning, and artificial intelligence. The content goes on
        to describe how these technologies are changing the world and
        creating new opportunities for developers and researchers alike.
        There are many applications being built using these technologies.
        """
        
        async with OllamaSummarizer(config) as summarizer:
            result = await summarizer.summarize_with_metadata(long_content)
            
            assert isinstance(result, SummaryResult)
            assert result.summary_length > 0
            assert result.original_length > 0

    @pytest.mark.asyncio
    async def test_extract_facts(self):
        """Test fact extraction."""
        config = LLMConfig(provider="ollama", max_tokens=300)
        
        content = """
        Python was created by Guido van Rossum in 1991.
        It is an interpreted, high-level programming language.
        Python emphasizes code readability and uses significant indentation.
        """
        
        async with OllamaSummarizer(config) as summarizer:
            facts = await summarizer.extract_facts(content)
            
            assert isinstance(facts, list)
            assert len(facts) > 0


@ollama_required
class TestMemorySummarizer:
    """Tests for high-level MemorySummarizer."""

    @pytest.mark.asyncio
    async def test_summarize_conversation(self):
        """Test conversation summarization."""
        messages = [
            {"role": "user", "content": "How do I sort a list in Python?"},
            {"role": "assistant", "content": "You can use sorted() or list.sort()"},
            {"role": "user", "content": "What's the difference?"},
            {"role": "assistant", "content": "sorted() returns a new list, sort() modifies in place"},
        ]
        
        async with MemorySummarizer() as summarizer:
            summary = await summarizer.summarize_conversation(messages)
            
            assert isinstance(summary, str)
            assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_extract_memory_content(self):
        """Test memory content extraction."""
        content = """
        The user mentioned they prefer using VS Code for Python development.
        They like the integrated terminal and the Python extension.
        """
        
        async with MemorySummarizer() as summarizer:
            result = await summarizer.extract_memory_content(content)
            
            assert "summary" in result
            assert isinstance(result["summary"], str)

    @pytest.mark.asyncio
    async def test_score_importance(self):
        """Test LLM-based importance scoring."""
        content = "I finally fixed that critical bug that's been blocking the release for weeks!"
        
        async with MemorySummarizer() as summarizer:
            scores = await summarizer.score_importance(content)
            
            assert "emotional_salience" in scores
            assert "novelty" in scores
            assert "causal_significance" in scores
            assert all(0 <= v <= 1 for v in scores.values())
