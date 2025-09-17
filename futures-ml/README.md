# Futures ML — Binance USDT‑M Trading Bot

Пълен проект от нулата с само Binance USDT‑M, Telegram, Docker Compose, Gate/Risk системи (Mark + Spread + L2 depth), почистване на „сираци", tick/lot guards, логове и backtest скеле. Моделът е TCN → BiLSTM → Transformer.

## Структура на проекта

```
futures-ml/
├─ .env
├─ docker-compose.yml
├─ requirements.txt
├─ config.yaml
├─ bot.log
├─ data/
│  ├─ fetch_ccxt.py
│  ├─ features.py
│  └─ labeling.py
├─ model/
│  ├─ tcn.py
│  ├─ hybrid_model.py
│  └─ losses.py
├─ train/
│  ├─ dataset.py
│  ├─ train.py
│  └─ walkforward.py
├─ live/
│  ├─ telegram.py
│  ├─ gate.py
│  ├─ risk.py
│  ├─ exchanges.py
│  ├─ orphan_cleaner.py
│  └─ run_inference.py
└─ backtest/
   ├─ engine.py
   └─ metrics.py
```

## Инсталация и стартиране

### 1. Конфигуриране на .env файла

Попълнете `.env` файла с вашите данни:

```bash
# Binance USDT‑M
BINANCE_KEY=your_actual_key
BINANCE_SECRET=your_actual_secret
BINANCE_TESTNET=false

# Символи (Binance USDT‑M формат)
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
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
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
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
python live/run_inference.py
```

## Контролен списък за Binance (важен)

1. ✅ USDT‑M Futures активирани; API ключът има Futures права
2. ✅ IP whitelist: изключен или добавен коректно
3. ✅ Position mode: One‑way (или адаптирай reduceOnly при Hedge)
4. ✅ Символи точно BTC/USDT (без :USDT)
5. ✅ Времето синхронизирано (NTP) – иначе recvWindow грешки
6. ✅ tick/lot guards: използваме price_to_precision/amount_to_precision преди ордер
7. ✅ Всички ордери се подават с SL/TP conditional поръчки

## Функционалности

- **ML Модел**: TCN → BiLSTM → Transformer архитектура
- **Risk Management**: Динамично изчисляване на позиции, TP/SL базирани на ATR
- **Gate Systems**: Price gate, spread guard, L2 depth проверки
- **Orphan Cleaner**: Автоматично почистване на "сираци" ордери
- **Telegram Integration**: Известия за всички търговски действия
- **Logging**: Пълно логване с ротация на файлове
- **Docker Support**: Лесно деплойване

## Забележки

- Моделът е инициализиран с случайни тежести (TODO: заредете обучен модел)
- Всички guard системи са активни за максимална сигурност
- Поддържа само Binance USDT-M futures
- Автоматично почистване на orphan ордери на всеки цикъл

## Разширения

Ако искате да добавите:
- COIN-M поддръжка
- Hedge mode
- News blackout
- Plotly табло

Кажете и ще ги вкарам в същия пакет.