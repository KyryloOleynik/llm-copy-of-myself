"""Telegram bot backed by the locally trained Qwen LoRA adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv(ENV_PATH)

TOKEN = os.getenv("BOT_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B")
ADAPTER_PATH = Path(
    os.getenv("ADAPTER_PATH", str(ROOT / "artifacts/training/qwen3-8b-r16/adapter-final"))
)
MIN_REPLY_DELAY = float(os.getenv("MIN_REPLY_DELAY", "5"))
MAX_REPLY_DELAY = float(os.getenv("MAX_REPLY_DELAY", "60"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))
STATE_DATABASE = Path(os.getenv("STATE_DATABASE", str(ROOT / "data/bot.sqlite3")))
RELATIONSHIPS = {
    "close_friend": "Close friend",
    "friend": "Friend",
    "acquaintance": "Acquaintance",
    "professional_contact": "Professional contact",
    "mother": "Mother",
    "father": "Father",
    "school_acquaintance": "School acquaintance",
}

dp = Dispatcher()
model: "LocalModel | None" = None
pending_chats: dict[int, "PendingChat"] = {}
chat_history: dict[int, list[dict[str, str]]] = {}
generation_queue: asyncio.Queue["GenerationRequest"] | None = None
user_relationships: dict[int, str] = {}


def relationship_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"relationship:{value}")]
            for value, label in RELATIONSHIPS.items()
        ]
    )


def load_relationships() -> None:
    STATE_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(STATE_DATABASE) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS user_relationships ("
            "user_id INTEGER PRIMARY KEY, relationship TEXT NOT NULL)"
        )
        rows = connection.execute("SELECT user_id, relationship FROM user_relationships")
        user_relationships.update(
            (user_id, relationship)
            for user_id, relationship in rows
            if relationship in RELATIONSHIPS
        )


def save_relationship(user_id: int, relationship: str) -> None:
    with sqlite3.connect(STATE_DATABASE) as connection:
        connection.execute(
            "INSERT INTO user_relationships (user_id, relationship) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET relationship = excluded.relationship",
            (user_id, relationship),
        )


class LocalModel:
    """A base model and trained adapter kept resident for the bot's lifetime."""

    def __init__(self, base_model: str, adapter_path: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Trained adapter not found: {adapter_path}")

        logging.info("Loading tokenizer and model into memory from %s", adapter_path)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        logging.info("Model loaded and ready; it will remain resident until bot.py exits")

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


@dataclass
class PendingChat:
    bot: Bot
    chat_id: int
    first_received: float
    last_received: float
    relationship: str
    messages: list[str] = field(default_factory=list)
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


@dataclass
class GenerationRequest:
    chat_id: int
    relationship: str
    incoming: str
    result: asyncio.Future[str]


async def generation_worker() -> None:
    """Serialize GPU inference and keep each chat's history in exact reply order."""
    assert generation_queue is not None
    assert model is not None
    while True:
        request = await generation_queue.get()
        try:
            if request.result.cancelled():
                continue
            history = chat_history.setdefault(request.chat_id, [])
            history.append({"role": "user", "content": request.incoming})
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI representation of Rodion. Respond in Rodion's learned "
                        "communication style. The user's relationship to Rodion is "
                        f"{RELATIONSHIPS[request.relationship]}. Adjust familiarity, tone, and "
                        "boundaries appropriately."
                    ),
                },
                *history[-MAX_HISTORY_TURNS * 2 :],
            ]
            reply = await asyncio.to_thread(model.generate, prompt)
            if not reply:
                reply = "I couldn't generate a reply this time."
            history.append({"role": "assistant", "content": reply})
            del history[: max(0, len(history) - MAX_HISTORY_TURNS * 2)]
            if not request.result.cancelled():
                request.result.set_result(reply)
        except Exception as exc:
            if not request.result.cancelled():
                request.result.set_exception(exc)
        finally:
            generation_queue.task_done()


async def wait_and_reply(pending: PendingChat) -> None:
    """Debounce a chat, but never wait over MAX_REPLY_DELAY in total."""
    try:
        while True:
            now = time.monotonic()
            send_at = min(
                pending.last_received + MIN_REPLY_DELAY,
                pending.first_received + MAX_REPLY_DELAY,
            )
            timeout = max(0.0, send_at - now)
            pending.wakeup.clear()
            try:
                await asyncio.wait_for(pending.wakeup.wait(), timeout=timeout)
                continue
            except TimeoutError:
                break

        # Remove this batch before inference so later arrivals start a new queue entry.
        if pending_chats.get(pending.chat_id) is pending:
            del pending_chats[pending.chat_id]

        assert generation_queue is not None
        result = asyncio.get_running_loop().create_future()
        await generation_queue.put(
            GenerationRequest(
                pending.chat_id,
                pending.relationship,
                "\n".join(pending.messages),
                result,
            )
        )
        reply = await result
        await pending.bot.send_message(pending.chat_id, reply)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Failed to generate a reply for chat %s", pending.chat_id)
        await pending.bot.send_message(pending.chat_id, "I couldn't generate a reply right now.")


def enqueue_message(message: Message, relationship: str) -> None:
    text = message.text or message.caption
    if not text:
        return

    now = time.monotonic()
    pending = pending_chats.get(message.chat.id)
    if pending is None:
        pending = PendingChat(message.bot, message.chat.id, now, now, relationship, [text])
        pending_chats[message.chat.id] = pending
        pending.task = asyncio.create_task(wait_and_reply(pending))
    else:
        pending.messages.append(text)
        pending.last_received = now
        pending.wakeup.set()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(
        "Hello! This is an AI representation of Rodion. It communicates in a similar "
        "style, but may not reflect Rodion's current thoughts or intentions.\n\n"
        "Before we speak, choose your relationship to Rodion:",
        reply_markup=relationship_keyboard(),
    )


@dp.message(Command("style", "status"))
async def style_handler(message: Message) -> None:
    await message.answer(
        "Choose your relationship to Rodion:", reply_markup=relationship_keyboard()
    )


@dp.message(Command("delete"))
async def delete_handler(message: Message) -> None:
    pending = pending_chats.pop(message.chat.id, None)
    if pending and pending.task:
        pending.task.cancel()
    chat_history.pop(message.chat.id, None)
    await message.answer(
        "Your stored conversation and queued messages have been deleted. "
        "Your selected response style was preserved."
    )


@dp.message(Command("about"))
async def about_handler(message: Message) -> None:
    await message.answer(
        "I am an AI representation of Rodion, powered by a locally running Qwen model "
        "with a QLoRA adapter trained on Rodion's communication style. I may not reflect "
        "Rodion's current thoughts, knowledge, or intentions."
    )


@dp.callback_query(lambda query: bool(query.data and query.data.startswith("relationship:")))
async def relationship_handler(query: CallbackQuery) -> None:
    relationship = query.data.split(":", 1)[1] if query.data else ""
    if relationship not in RELATIONSHIPS:
        await query.answer("Invalid relationship status.", show_alert=True)
        return
    user_relationships[query.from_user.id] = relationship
    save_relationship(query.from_user.id, relationship)
    await query.answer("Status saved.")
    if query.message:
        await query.message.edit_text(
            f"Status saved: {RELATIONSHIPS[relationship]}. You can speak now. "
            "Use /style to change it later."
        )


@dp.message()
async def reply_handler(message: Message) -> None:
    """Queue text/caption messages instead of echoing or replying immediately."""
    if message.from_user is None:
        return
    relationship = user_relationships.get(message.from_user.id)
    if relationship is None:
        await message.answer(
            "Before we speak, choose your relationship to Rodion:",
            reply_markup=relationship_keyboard(),
        )
        return
    enqueue_message(message, relationship)


async def main() -> None:
    global generation_queue, model
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in the environment or .env file")
    if MIN_REPLY_DELAY < 0 or MAX_REPLY_DELAY < MIN_REPLY_DELAY:
        raise ValueError("Reply delays must satisfy 0 <= MIN_REPLY_DELAY <= MAX_REPLY_DELAY")

    load_relationships()
    # Load synchronously before polling: no request can arrive before the model is ready.
    model = LocalModel(BASE_MODEL, ADAPTER_PATH)
    generation_queue = asyncio.Queue()
    worker = asyncio.create_task(generation_worker())
    bot = Bot(token=TOKEN)
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Start a conversation"),
                BotCommand(command="style", description="Adjust the response style"),
                BotCommand(command="delete", description="Delete stored conversation data"),
                BotCommand(command="about", description="Learn about Rodion AI"),
            ]
        )
        await bot.set_my_description(
            "An AI representation that communicates in a style similar to Rodion. "
            "It may not reflect Rodion's current thoughts or intentions."
        )
        await bot.set_my_short_description("An AI bot that communicates like Rodion.")
        logging.info("Starting bot polling")
        await dp.start_polling(bot)
    finally:
        for pending in pending_chats.values():
            if pending.task:
                pending.task.cancel()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
