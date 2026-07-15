"""Telegram bot backed by the locally trained Qwen LoRA adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender
from personal_ai.modeling import (
    generate_reply,
    load_inference_model,
    personal_style_generation_options,
)
from personal_ai.utils import load_dotenv, read_json, relationship_system_message, render_chat_ids


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


load_dotenv(ENV_PATH)

TOKEN = os.getenv("BOT_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ADAPTER_PATH = Path(
    os.getenv(
        "ADAPTER_PATH",
        str(ROOT / "artifacts/training/qwen3-4b-instruct-2507-r16/adapter-final"),
    )
)
MIN_REPLY_DELAY = float(os.getenv("MIN_REPLY_DELAY", "2"))
MAX_REPLY_DELAY = float(os.getenv("MAX_REPLY_DELAY", "60"))
READING_CHARS_PER_SECOND = float(os.getenv("READING_CHARS_PER_SECOND", "35"))
TYPING_WORDS_PER_MINUTE = float(os.getenv("TYPING_WORDS_PER_MINUTE", "42"))
MAX_POST_GENERATION_DELAY = float(os.getenv("MAX_POST_GENERATION_DELAY", "5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
MAX_REPLY_PARTS = int(os.getenv("MAX_REPLY_PARTS", "4"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "30"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
STATE_DATABASE = Path(os.getenv("STATE_DATABASE", str(ROOT / "data/bot.sqlite3")))
EVALUATION_REPORT = Path(
    os.getenv("EVALUATION_REPORT", str(ROOT / "data/processed/evaluation/evaluation.json"))
)
PREPARED_MANIFEST = Path(
    os.getenv("PREPARED_MANIFEST", str(ROOT / "data/processed/manifest.json"))
)
RELATIONSHIPS = {
    "close_friend": "Close friend",
    "friend": "Friend",
    "acquaintance": "Acquaintance",
    "professional_contact": "Professional contact",
    "family": "Family",
    "school_acquaintance": "School acquaintance",
}

dp = Dispatcher()
model: "LocalModel | None" = None
pending_chats: dict[int, "PendingChat"] = {}
chat_history: dict[int, list[dict[str, str]]] = {}
generation_queue: asyncio.Queue["GenerationRequest"] | None = None
user_relationships: dict[int, str] = {}


def _has_adapter_weights(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _is_complete_checkpoint(path: Path) -> bool:
    return _has_adapter_weights(path) and all(
        (path / name).is_file()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    )


def resolve_adapter_path(preferred: Path) -> Path:
    """Use adapter-final when available, otherwise the newest complete checkpoint."""
    if _has_adapter_weights(preferred):
        return preferred
    if preferred.name != "adapter-final":
        raise FileNotFoundError(f"Trained adapter not found or incomplete: {preferred}")

    checkpoints: list[tuple[int, Path]] = []
    for path in preferred.parent.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and _is_complete_checkpoint(path):
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(
            f"Final adapter not found and no complete checkpoints exist in {preferred.parent}"
        )
    checkpoint = max(checkpoints, key=lambda item: item[0])[1]
    logging.warning("Final adapter is unavailable; loading latest checkpoint: %s", checkpoint)
    return checkpoint


def require_adapter_dataset_match(adapter_path: Path, manifest_path: Path) -> None:
    """Refuse to run an adapter trained against a different prepared dataset."""
    metadata_path = adapter_path.parent / "reproducibility.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("Prepared dataset or adapter reproducibility metadata is missing")
    dataset_hash = read_json(manifest_path).get("dataset_sha256")
    trained_hash = read_json(metadata_path).get("dataset_sha256")
    if not dataset_hash or trained_hash != dataset_hash:
        raise RuntimeError(
            "Adapter was trained on a different dataset; run smoke training and full "
            "training again with --fresh"
        )


def natural_response_delay(incoming: str, reply: str) -> float:
    """Estimate how long a person would need to read and compose this reply."""
    reading_seconds = min(6.0, max(0.8, len(incoming) / READING_CHARS_PER_SECOND))
    reply_words = max(1, len(reply.split()))
    typing_seconds = min(12.0, reply_words * 60.0 / TYPING_WORDS_PER_MINUTE)
    variation = random.uniform(0.85, 1.15)
    return (reading_seconds + typing_seconds) * variation


def between_message_delay(message: str) -> float:
    """Pause briefly between separately sent lines without making long replies tedious."""
    return min(2.0, max(0.45, len(message) / 80.0)) * random.uniform(0.85, 1.15)


def safe_reply_parts(reply: str) -> list[str]:
    """Deduplicate generated lines and cap Telegram messages from one model response."""
    parts: list[str] = []
    seen: set[str] = set()
    for line in reply.splitlines():
        part = line.strip()
        normalized = part.casefold()
        if not part or normalized in seen:
            continue
        seen.add(normalized)
        parts.append(part)
        if len(parts) == MAX_REPLY_PARTS:
            break
    return parts


def incoming_message_content(message: Message) -> str | None:
    """Represent Telegram media as text markers the language model understands."""
    media_checks = (
        ("photo", "[sent image]"),
        ("video", "[sent video]"),
        ("video_note", "[sent video]"),
        ("animation", "[sent animation]"),
        ("voice", "[sent voice message]"),
        ("audio", "[sent audio file]"),
        ("document", "[sent document]"),
        ("sticker", "[sent sticker]"),
        ("location", "[sent location]"),
        ("venue", "[sent location]"),
        ("contact", "[sent contact]"),
        ("poll", "[sent poll]"),
        ("dice", "[sent dice]"),
    )
    marker = next(
        (placeholder for attribute, placeholder in media_checks if getattr(message, attribute, None)),
        None,
    )
    if marker:
        caption = (message.caption or "").strip()
        return f"{marker}\n{caption}" if caption else marker
    text = (message.text or message.caption or "").strip()
    return text or None


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


def confirm_unverified_adapter(message: str) -> bool:
    """Require explicit terminal confirmation before bypassing a failed evaluation gate."""
    if not sys.stdin.isatty():
        return False
    print(f"WARNING: {message}", file=sys.stderr)
    try:
        answer = input("Continue with this unverified adapter? [y/N]: ").strip().casefold()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False
    return answer in {"y", "yes"}


class LocalModel:
    """A base model and trained adapter kept resident for the bot's lifetime."""

    def __init__(self, base_model: str, adapter_path: Path) -> None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Trained adapter not found: {adapter_path}")
        require_adapter_dataset_match(adapter_path, PREPARED_MANIFEST)
        if not EVALUATION_REPORT.is_file():
            raise RuntimeError(
                f"Adapter acceptance report is missing: {EVALUATION_REPORT}. "
                "Run personal-ai evaluate and complete blind style ratings first."
            )
        evaluation = read_json(EVALUATION_REPORT)
        result = evaluation.get("results", {}).get(adapter_path.name, {})
        if not result.get("accepted"):
            gate_error = (
                f"Adapter {adapter_path.name} has not passed the evaluation gate: "
                f"{result.get('acceptance_reasons', ['candidate not found'])}"
            )
            if not confirm_unverified_adapter(gate_error):
                raise RuntimeError(gate_error)
            logging.warning("Evaluation gate bypassed by explicit console confirmation")

        logging.info("Loading tokenizer and model into memory from %s", adapter_path)
        self.torch, self.tokenizer, self.model = load_inference_model(
            base_model, adapter_path, "auto"
        )
        logging.info("Model loaded and ready; it will remain resident until bot.py exits")

    def _fit_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop oldest complete turns until prompt plus reply reserve fits 8K."""
        fitted = list(messages)
        while True:
            input_ids = render_chat_ids(self.tokenizer, fitted, generation=True)
            if len(input_ids) + MAX_NEW_TOKENS <= MAX_CONTEXT_TOKENS:
                return fitted
            if len(fitted) <= 2:
                raise ValueError("System prompt and latest user message exceed context budget")
            fitted.pop(1)
            if len(fitted) > 2 and fitted[1]["role"] == "assistant":
                fitted.pop(1)

    def generate(self, messages: list[dict[str, str]]) -> str:
        messages = self._fit_messages(messages)
        reply, _ = generate_reply(
            self.torch,
            self.tokenizer,
            self.model,
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            **personal_style_generation_options(),
        )
        return reply


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
                    "content": relationship_system_message(request.relationship),
                    "role": "system",
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

        incoming = "\n".join(pending.messages)
        started_at = time.monotonic()
        async with ChatActionSender(
            bot=pending.bot,
            chat_id=pending.chat_id,
            action=ChatAction.TYPING,
            interval=4.5,
        ):
            assert generation_queue is not None
            result = asyncio.get_running_loop().create_future()
            await generation_queue.put(
                GenerationRequest(
                    pending.chat_id,
                    pending.relationship,
                    incoming,
                    result,
                )
            )
            reply = await result
            elapsed = time.monotonic() - started_at
            remaining_delay = max(0.0, natural_response_delay(incoming, reply) - elapsed)
            await asyncio.sleep(min(remaining_delay, MAX_POST_GENERATION_DELAY))

        parts = safe_reply_parts(reply)
        if not parts:
            parts = ["I couldn't generate a coherent reply right now."]
        for index, part in enumerate(parts):
            if index:
                await pending.bot.send_chat_action(pending.chat_id, ChatAction.TYPING)
                await asyncio.sleep(between_message_delay(part))
            await pending.bot.send_message(pending.chat_id, part)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Failed to generate a reply for chat %s", pending.chat_id)
        await pending.bot.send_message(pending.chat_id, "I couldn't generate a reply right now.")


def enqueue_message(message: Message, relationship: str) -> None:
    content = incoming_message_content(message)
    if not content:
        return

    now = time.monotonic()
    pending = pending_chats.get(message.chat.id)
    if pending is None:
        pending = PendingChat(message.bot, message.chat.id, now, now, relationship, [content])
        pending_chats[message.chat.id] = pending
        pending.task = asyncio.create_task(wait_and_reply(pending))
    else:
        pending.messages.append(content)
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
    """Queue text and normalized media placeholders for the local language model."""
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
    if READING_CHARS_PER_SECOND <= 0 or TYPING_WORDS_PER_MINUTE <= 0:
        raise ValueError("Reading speed and typing speed must be greater than zero")
    if MAX_POST_GENERATION_DELAY < 0:
        raise ValueError("MAX_POST_GENERATION_DELAY must be non-negative")
    if MAX_REPLY_PARTS < 1:
        raise ValueError("MAX_REPLY_PARTS must be at least one")

    load_relationships()
    # Load synchronously before polling: no request can arrive before the model is ready.
    model = LocalModel(BASE_MODEL, resolve_adapter_path(ADAPTER_PATH))
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
