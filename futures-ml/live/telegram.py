from os import getenv
from telegram import Bot
from loguru import logger

class TGBot:
    def __init__(self):
        self.token = getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = getenv("TELEGRAM_CHAT_ID")
        self.bot = Bot(self.token) if self.token else None
    async def send(self, text: str):
        if not self.bot or not self.chat_id:
            logger.warning("Telegram not configured")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"TG error: {e}")