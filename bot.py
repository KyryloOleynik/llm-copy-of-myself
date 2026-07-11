import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

# Load environment variables from .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Retrieve Bot Token from environment
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    logging.critical("BOT_TOKEN is not set in environment or .env file.")
    sys.exit(1)

# All handlers should be attached to the Dispatcher
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n\n"
        f"I am a basic aiogram bot template. Send me any message, and I will echo it back!"
    )


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Handler will forward the received message back to the sender.
    By default, it handles all message types (text, photo, sticker, etc.)
    """
    try:
        # Send a copy of the received message
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        # If type cannot be copied (e.g. some complex media types)
        await message.answer("Nice try, but I cannot copy this message type!")


async def main() -> None:
    # Initialize Bot instance with default HTML parsing mode
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    logging.info("Starting bot...")
    # Run the event dispatcher
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
