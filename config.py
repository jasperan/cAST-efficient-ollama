import os
import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self._load_config()

    def _load_config(self):
        config_file = Path('config.yaml')
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Oracle
            self.ORACLE_USER = config_data.get('oracle', {}).get('user', 'scott')
            self.ORACLE_PASSWORD = config_data.get('oracle', {}).get('password', 'password')
            self.ORACLE_DSN = config_data.get('oracle', {}).get('dsn', 'localhost:1521/freepdb1')
            
            # Ollama
            self.OLLAMA_ENDPOINT = config_data.get('ollama', {}).get('endpoint', 'http://localhost:11434')
            self.OLLAMA_MODEL = config_data.get('ollama', {}).get('model', 'llama2')
            
            # App
            self.VERBOSE = config_data.get('app', {}).get('verbose', False)
            self.CHUNK_SIZE = int(config_data.get('app', {}).get('chunk_size', 2000))
            self.OVERLAP_PERCENTAGE = int(config_data.get('app', {}).get('overlap_percentage', 10))
            self.TOP_K_RETRIEVAL = int(config_data.get('app', {}).get('top_k_retrieval', 150))
            self.TOP_K_RERANK = int(config_data.get('app', {}).get('top_k_rerank', 20))
        else:
            # Fallback to environment variables
            self.ORACLE_USER = os.getenv('ORACLE_USER', 'scott')
            self.ORACLE_PASSWORD = os.getenv('ORACLE_PASSWORD', 'password')
            self.ORACLE_DSN = os.getenv('ORACLE_DSN', 'localhost:1521/freepdb1')
            
            self.OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
            self.OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
            
            self.VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'
            self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 2000))
            self.OVERLAP_PERCENTAGE = int(os.getenv('OVERLAP_PERCENTAGE', 10))
            self.TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', 150))
            self.TOP_K_RERANK = int(os.getenv('TOP_K_RERANK', 20))

# Instantiate
Config = Config()

# Usage: from config import Config
# print(Config.ORACLE_USER)
