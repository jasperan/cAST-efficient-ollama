import oracledb
from config import Config
import logging

logger = logging.getLogger(__name__)

class DBConnection:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnection, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        try:
            self._pool = oracledb.create_pool(
                user=Config.ORACLE_USER,
                password=Config.ORACLE_PASSWORD,
                dsn=Config.ORACLE_DSN,
                min=2,
                max=5,
                increment=1
            )
            logger.info("Oracle DB connection pool initialized successfully.")
        except oracledb.Error as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise

    def get_connection(self):
        try:
            conn = self._pool.acquire()
            return conn
        except oracledb.Error as e:
            logger.error(f"Error acquiring connection: {e}")
            # Attempt to reinitialize pool on failure
            self._initialize_pool()
            try:
                conn = self._pool.acquire()
                return conn
            except oracledb.Error as retry_e:
                logger.error(f"Retry failed: {retry_e}")
                raise

    def close_pool(self):
        if self._pool:
            self._pool.close()
            logger.info("Oracle DB connection pool closed.")
