from pydantic import BaseModel
from typing import Optional, List
from ollama import chat
from config import Config
import logging
import hashlib

logger = logging.getLogger(__name__)

class CodeMetadata(BaseModel):
    purpose: str
    input_params: str
    return_type: str
    docstring: Optional[str]
    dependencies: List[str]
    complexity: str  # 'low', 'medium', 'high'

class OllamaMetadataExtractor:
    def __init__(self, endpoint: str = Config.OLLAMA_ENDPOINT, model: str = Config.OLLAMA_MODEL):
        self.endpoint = endpoint
        self.model = model
        self.cache = {}

    def extract(self, code_chunk: str) -> CodeMetadata:
        """
        Extract metadata from code chunk using Ollama with structured outputs.
        Uses Pydantic schema for guaranteed JSON structure.
        """
        # Check cache first
        chunk_hash = hashlib.sha256(code_chunk.encode()).hexdigest()
        if chunk_hash in self.cache:
            return self.cache[chunk_hash]

        prompt = f"""
        Analyze this Python code and extract metadata in JSON format.

        Code:
        {code_chunk}

        Extract:
        1. purpose: One-line description of what this code does
        2. input_params: Description of input parameters
        3. return_type: What this function/class returns
        4. docstring: A clean docstring (optional)
        5. dependencies: List of imported modules or called functions
        6. complexity: 'low', 'medium', or 'high' based on code complexity

        Return only valid JSON matching the schema.
        """

        try:
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=CodeMetadata.model_json_schema(),
                stream=False,
                options={"temperature": 0.0}  # For deterministic output
            )

            metadata = CodeMetadata.model_validate_json(response['message']['content'])
            self.cache[chunk_hash] = metadata
            logger.info("Metadata extracted successfully.")
            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return defaults on failure
            return CodeMetadata(
                purpose="Unable to extract - check Ollama connection",
                input_params="",
                return_type="unknown",
                docstring=None,
                dependencies=[],
                complexity="medium"
            )
