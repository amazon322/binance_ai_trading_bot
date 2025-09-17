# Futures ML Bot - Binance USDT-M

Пълен проект от нулата за търговия с фючърси на Binance USDT-M с използване на машинно обучение.

## Структура на проекта

```
futures-ml/
├─ .env                    # Конфигурационни настройки
├─ docker-compose.yml      # Docker конфигурация
├─ requirements.txt       # Python зависимости
├─ config.yaml           # YAML конфигурация
├─ bot.log               # Логове
├─ data/                 # Модул за данни
│  ├─ fetch_ccxt.py     # Извличане на данни от Binance
│  ├─ features.py       # Технически индикатори
│  └─ labeling.py       # Triple barrier labeling
├─ model/                # ML модели
│  ├─ tcn.py           # Temporal Convolutional Network
│  ├─ hybrid_model.py  # TCN → BiLSTM → Transformer
│  └─ losses.py        # Focal Loss
├─ train/               # Обучение на модели
│  ├─ dataset.py       # Dataset utilities
│  ├─ train.py         # Тренировъчна логика
│  └─ walkforward.py   # Walk-forward validation
├─ live/                # Live търговия
│  ├─ telegram.py      # Telegram уведомления
│  ├─ gate.py          # Risk gates
│  ├─ risk.py          # Risk management
│  ├─ exchanges.py     # Binance broker
│  ├─ orphan_cleaner.py # Почистване на "сираци"
│  └─ run_inference.py # Главен файл за търговия
└─ backtest/           # Backtesting
   ├─ engine.py        # Backtest engine
   └─ metrics.py       # Метрики за оценка
```

## Модел

Хибриден модел: **TCN → BiLSTM → Transformer**
- TCN за извличане на временни зависимости
- BiLSTM за последователно моделиране
- Transformer за внимание и финална класификация/регресия

## Функции

- ✅ Само Binance USDT-M фючърси
- ✅ Telegram уведомления
- ✅ Docker Compose за лесно стартиране
- ✅ Gate/Risk система (Mark + Spread + L2 depth)
- ✅ Почистване на "сираци" ордери
- ✅ Tick/lot guards
- ✅ Логове с loguru
- ✅ Backtest скеле

## Стартиране

### 1. Конфигуриране

Попълнете `.env` файла с вашите настройки:

```bash
# Binance USDT‑M
BINANCE_KEY=your_key
BINANCE_SECRET=your_secret
BINANCE_TESTNET=false

# Символи
SYMBOLS=BTC/USDT,ETH/USDT
TIMEFRAME=5m
LOOKBACK=128

# Риск и guard‑ове
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

### 2. Стартиране с Docker

```bash
docker compose up --build
```

### 3. Стартиране локално

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
python live/run_inference.py
```

## Контролен списък за Binance

1. ✅ USDT‑M Futures активирани; API ключът има Futures права
2. ✅ IP whitelist: изключен или добавен коректно
3. ✅ Position mode: One‑way (или адаптирай reduceOnly при Hedge)
4. ✅ Символи точно BTC/USDT (без :USDT)
5. ✅ Времето синхронизирано (NTP) – иначе recvWindow грешки
6. ✅ Tick/lot guards: използваме price_to_precision/amount_to_precision преди ордер
7. ✅ Всички ордери се подават с SL/TP conditional поръчки

## Предупреждение

Този бот е за образователни цели. Винаги тествайте с малки суми и testnet преди да използвате реални пари. Търговията с фючърси е рискова и може да доведе до загуби.