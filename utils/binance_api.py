import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from config import Config

class BinanceAPI:
    def __init__(self):
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_SECRET_KEY
        self.client = Client(api_key, api_secret)

    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        try:
            klines = self.client.get_historical_klines(
                symbol,
                interval,
                start_str,
                end_str
            )
            return klines
        except BinanceAPIException as e:
            # Handle API exceptions
            print(f"Binance API Exception: {e}")
            return None
        except Exception as e:
            # Handle general exceptions
            print(f"An error occurred: {e}")
            return None

    def place_order(self, symbol, side, order_type, quantity, price=None, stop_price=None, time_in_force='GTC'):
        try:
            if order_type == 'MARKET':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity
                )
            elif order_type == 'LIMIT':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    timeInForce=time_in_force,
                    quantity=quantity,
                    price=str(price)
                )
            elif order_type in ['STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    timeInForce=time_in_force,
                    quantity=quantity,
                    price=str(price),
                    stopPrice=str(stop_price)
                )
            else:
                print("Unsupported order type")
                return None
            return order
        except BinanceAPIException as e:
            # Handle API exceptions
            print(f"Binance API Exception: {e}")
            return None
        except BinanceOrderException as e:
            # Handle order exceptions
            print(f"Binance Order Exception: {e}")
            return None
        except Exception as e:
            # Handle general exceptions
            print(f"An error occurred: {e}")
            return None

    def get_account_info(self):
        try:
            account_info = self.client.get_account()
            return account_info
        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_symbol_ticker(self, symbol):
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return ticker
        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_current_average_price(self, symbol):
        try:
            avg_price = self.client.get_avg_price(symbol=symbol)
            return avg_price
        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_exchange_info(self):
        try:
            exchange_info = self.client.get_exchange_info()
            return exchange_info
        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            return None

    def cancel_order(self, symbol, order_id):
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            return result
        except BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            return None
