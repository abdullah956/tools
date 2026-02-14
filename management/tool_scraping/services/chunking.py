"""Text chunking utilities using LangChain MarkdownTextSplitter."""

import logging

from langchain.text_splitter import MarkdownTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Text chunking service that respects markdown structure.

    Uses MarkdownTextSplitter to split text while preserving markdown structure
    (headers, lists, code blocks, etc.) at the target chunk size.
    """

    def __init__(self):
        """Initialize the TextChunker with MarkdownTextSplitter."""
        # Markdown splitter will be initialized with target chunk size
        self.markdown_splitter = None

    def chunk(self, text, chunk_size=1000, overlap=50):
        """
        Chunks text using MarkdownTextSplitter directly.

        Respects markdown structure (headers, lists, code blocks, etc.)
        and splits at the target chunk size.

        Args:
            text (str): Text to chunk
            chunk_size (int): Target chunk size in characters (default 1000)
            overlap (int): Overlap between chunks (default 50)

        Returns:
            list: List of text chunks split by markdown structure
        """
        try:
            if not text or len(text) == 0:
                logger.warning("Empty text provided for chunking")
                return []

            logger.info(
                f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}"
            )

            # Initialize splitter with target chunk size
            self.markdown_splitter = MarkdownTextSplitter(
                chunk_size=chunk_size, chunk_overlap=overlap
            )

            # Direct markdown splitting
            chunks = self.markdown_splitter.split_text(text)

            logger.info(f"Created {len(chunks)} chunks using MarkdownTextSplitter")

            # Log chunk size statistics
            if chunks:
                sizes = [len(chunk) for chunk in chunks]
                logger.debug(
                    f"Chunk sizes - min: {min(sizes)}, max: {max(sizes)}, "
                    f"avg: {sum(sizes) / len(sizes):.0f}"
                )

            return chunks

        except Exception as e:
            logger.exception(f"Error chunking text: {e}")
            return []
