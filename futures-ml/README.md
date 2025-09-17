# Futures ML Bot - Binance USDT-M

Пълен проект от нулата за автоматизирано търгуване с Binance USDT-M фючърси, използващ ML модел TCN → BiLSTM → Transformer.

## Структура на проекта

```
futures-ml/
├─ .env                     # Конфигурация (API ключове, параметри)
├─ docker-compose.yml       # Docker контейнер
├─ requirements.txt         # Python зависимости
├─ config.yaml             # YAML конфигурация
├─ data/                   # Обработка на данни
│  ├─ fetch_ccxt.py        # Извличане на данни от Binance
│  ├─ features.py          # Технически индикатори
│  └─ labeling.py          # Triple barrier labeling
├─ model/                  # ML модели
│  ├─ tcn.py              # Temporal Convolutional Network
│  ├─ hybrid_model.py     # TCN + BiLSTM + Transformer
│  └─ losses.py           # Focal Loss функция
├─ train/                  # Обучение на модела
│  ├─ dataset.py          # Dataset клас
│  ├─ train.py            # Обучение
│  └─ walkforward.py      # Walk-forward анализ
├─ live/                   # Живо търгуване
│  ├─ telegram.py         # Telegram уведомления
│  ├─ gate.py             # Gate и Risk контроли
│  ├─ risk.py             # Risk management
│  ├─ exchanges.py        # BinanceUSDM брокер
│  ├─ orphan_cleaner.py   # Почистване на "сираци"
│  └─ run_inference.py    # Главен скрипт
└─ backtest/              # Backtesting
   ├─ engine.py           # Backtest движок
   └─ metrics.py          # Метрики за производителност
```

## Настройка

### 1. Попълни .env файла

```bash
# Binance USDT‑M API
BINANCE_KEY=your_api_key_here
BINANCE_SECRET=your_secret_key_here
BINANCE_TESTNET=false

# Символи за търгуване
SYMBOLS=BTC/USDT,ETH/USDT
TIMEFRAME=5m
LOOKBACK=128

# Risk management
RISK_PCT=0.005
SPREAD_GUARD_PCT=0.25
PRICE_GATE_PCT=0.3
DEPTH_WINDOW_PCT=0.3
MAX_OPEN_TRADES=10
DEDUP_HOURS=24

# Telegram уведомления
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
# или .venv\Scripts\activate  # Windows

pip install -r requirements.txt
python live/run_inference.py
```

## Важни настройки за Binance

1. **USDT‑M Futures активирани** - API ключът трябва да има Futures права
2. **IP whitelist** - изключен или добавен коректно
3. **Position mode** - One‑way (или адаптирай reduceOnly при Hedge)
4. **Символи** - точно BTC/USDT (без :USDT)
5. **Време синхронизация** - NTP синхронизирано, иначе recvWindow грешки
6. **Tick/lot guards** - автоматично използва price_to_precision/amount_to_precision

## Функционалности

### ML Модел
- **TCN** - за извличане на временни зависимости
- **BiLSTM** - за последователно моделиране
- **Transformer** - за attention механизми
- **Focal Loss** - за класово дисбаланс

### Risk Management
- **Position sizing** - базирано на ATR и риск процент
- **Dynamic TP/SL** - автоматично изчисляване
- **Spread guard** - контрол на ликвидност
- **Price gate** - контрол на цена отклонение
- **Depth guard** - контрол на order book дълбочина

### Система за почистване
- **Orphan cleaner** - автоматично почистване на "сираци" поръчки
- **Deduplication** - предотвратяване на дублирани сигнали
- **Max positions** - ограничаване на броя отворени позиции

### Мониторинг
- **Telegram уведомления** - за всяка търговска операция
- **Loguru логиране** - детайлни логове с ротация
- **Real-time monitoring** - непрекъснат мониторинг на пазара

## Безопасност

- Всички API ключове се съхраняват в .env файла
- Автоматично почистване на "сираци" поръчки
- Множество gate-ове за предотвратяване на грешки
- Testnet режим за тестване

## Разширения

Проектът може лесно да бъде разширен с:
- COIN-M фючърси поддръжка
- Hedge mode
- News blackout периоди
- Plotly dashboard
- Допълнителни биржи
- Други ML модели