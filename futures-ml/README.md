# Futures ML — Binance USDT‑M Bot

Пълен проект от нулата за автоматизиран търговски бот с ML модел TCN → BiLSTM → Transformer за Binance USDT‑M фючърси.

## Структура на проекта

```
futures-ml/
├─ .env                    # Конфигурация и API ключове
├─ docker-compose.yml      # Docker контейнер
├─ requirements.txt        # Python зависимости
├─ config.yaml            # YAML конфигурация
├─ bot.log               # Логове на бота
├─ data/                 # Модул за данни
│  ├─ fetch_ccxt.py      # Извличане на данни от Binance
│  ├─ features.py        # Технически индикатори
│  └─ labeling.py        # Triple barrier labeling
├─ model/                # ML модели
│  ├─ tcn.py            # Temporal Convolutional Network
│  ├─ hybrid_model.py   # TCN → BiLSTM → Transformer
│  └─ losses.py         # Focal Loss
├─ train/               # Обучение на модели
│  ├─ dataset.py       # Dataset utilities (placeholder)
│  ├─ train.py         # Training loop
│  └─ walkforward.py   # Walk-forward analysis (placeholder)
├─ live/               # Live търговия
│  ├─ telegram.py     # Telegram уведомления
│  ├─ gate.py         # Gate/Risk guards
│  ├─ risk.py         # Risk management
│  ├─ exchanges.py    # BinanceUSDM брокер + guards
│  ├─ orphan_cleaner.py # Почистване на "сираци"
│  └─ run_inference.py # Главен скрипт
└─ backtest/           # Backtesting
   ├─ engine.py       # Backtest engine
   └─ metrics.py      # Performance metrics (placeholder)
```

## Функционалности

- **ML Модел**: TCN → BiLSTM → Transformer архитектура
- **Gate/Risk**: Mark price gate, spread guard, L2 depth guard
- **Risk Management**: Динамичен TP/SL базиран на ATR
- **Orphan Cleaner**: Автоматично почистване на "сираци" ордери
- **Telegram**: Уведомления за търговски сигнали
- **Docker**: Готов за deployment
- **Logging**: Подробни логове с loguru

## Стартиране

### 1. Конфигурация

Попълни `.env` файла:

```bash
# Binance USDT‑M
BINANCE_KEY=your_actual_key
BINANCE_SECRET=your_actual_secret
BINANCE_TESTNET=false

# Символи
SYMBOLS=BTC/USDT,ETH/USDT
TIMEFRAME=5m
LOOKBACK=128

# Risk настройки
RISK_PCT=0.005
SPREAD_GUARD_PCT=0.25
PRICE_GATE_PCT=0.3
DEPTH_WINDOW_PCT=0.3
MAX_OPEN_TRADES=10
DEDUP_HOURS=24

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 2. Docker (препоръчително)

```bash
docker compose up --build
```

### 3. Локално

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
python live/run_inference.py
```

## Важни забележки за Binance

1. **USDT‑M Futures**: Активирани в акаунта
2. **API права**: Futures права за API ключа
3. **IP whitelist**: Изключен или добавен коректно
4. **Position mode**: One‑way (или адаптирай reduceOnly при Hedge)
5. **Символи**: Точно BTC/USDT (без :USDT)
6. **Време**: Синхронизирано (NTP) – иначе recvWindow грешки
7. **Precision**: Използва price_to_precision/amount_to_precision

## Модел

Моделът използва:
- **TCN**: За извличане на временни зависимости
- **BiLSTM**: За последователни модели
- **Transformer**: За attention механизъм
- **Dual head**: Classification + Regression за по-точно предсказване

## Risk Management

- **Position sizing**: Базиран на риск процент и stop loss
- **Dynamic TP/SL**: Използва ATR за адаптивни нива
- **Gate guards**: Проверки за ликвидност и цена
- **Orphan cleanup**: Автоматично почистване на загубени ордери

## Логове

Логовете се записват в `bot.log` с ротация на 5MB и задържане за 7 дни.

## Разширения

За добавяне на:
- COIN‑M фючърси
- Hedge mode
- News blackout
- Plotly dashboard

Свържи се за имплементация.