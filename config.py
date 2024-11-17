import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Binance API keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

    # Default settings
    DEFAULT_MODEL = 'LSTM'
    DEFAULT_COIN = 'BTCUSDT'
    DEFAULT_DATA_PERIOD = 30  # days
    MODEL_SAVE_PATH = 'saved_models/'

    # Continuous learning settings
    ENABLE_CONTINUOUS_LEARNING = False
    LEARNING_INTERVAL = 7  # days

    # Risk management settings
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    TRAILING_STOP = True
    TAKE_PROFIT_LEVEL = 0.2  # 20%

    # Scheduler settings
    SCHEDULER_INTERVAL = 24  # hours

    # Alerts and notifications
    ALERT_EMAIL = os.getenv('ALERT_EMAIL')
    ALERT_SMS = os.getenv('ALERT_SMS')

    # Other settings...
