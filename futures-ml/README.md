# Futures ML — Binance USDT‑M (Full v3)

Пълен проект от нулата, с само Binance USDT‑M, Telegram, Docker Compose, Gate/Risk (Mark + Spread + L2 depth), почистване на „сираци", tick/lot guards, логове и backtest скеле. Моделът е TCN → BiLSTM → Transformer.

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
│  ├─ dataset.py (placeholder)
│  ├─ train.py
│  └─ walkforward.py (placeholder)
├─ live/
│  ├─ telegram.py
│  ├─ gate.py
│  ├─ risk.py
│  ├─ exchanges.py  # BinanceUSDM брокер + guards
│  ├─ orphan_cleaner.py
│  └─ run_inference.py
└─ backtest/
   ├─ engine.py
   └─ metrics.py (placeholder)
```

## Как да стартираш

### Docker:
```bash
docker compose up --build
```

### Локално:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python live/run_inference.py
```

## Конфигурация

1. Попълни `.env` файла с твоите Binance API ключове и Telegram настройки
2. Настрой символите, риск параметрите и други настройки в `.env`
3. Стартирай бота

## Контролен списък за Binance (важен)

1. USDT‑M Futures активирани; API ключът има Futures права.
2. IP whitelist: изключен или добавен коректно.
3. Position mode: One‑way (или адаптирай reduceOnly при Hedge).
4. Символи точно BTC/USDT (без :USDT).
5. Времето синхронизирано (NTP) – иначе recvWindow грешки.
6. tick/lot guards: използваме price_to_precision/amount_to_precision преди ордер.
7. Всички ордери се подават с SL/TP conditional поръчки.

## Модел

Хибриден модел TCN → BiLSTM → Transformer за класификация и регресия на ценови движения.

## Риск мениджмънт

- Динамичен TP/SL базиран на ATR
- Position sizing базиран на риск процент
- Gate guards за цена, спред и дълбочина на пазара
- Почистване на orphan ордери
- Дедупликация на сигнали

## Логове

Всички логове се записват в `bot.log` с ротация на 5MB и задържане на 7 дни.