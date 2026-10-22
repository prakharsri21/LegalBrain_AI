"""FastAPI backend for LegalBrain AI."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field

from app.services.guardrails import (
    DISCLAIMER,
    SYSTEM_PROMPT_VERSION,
    build_messages,
    extract_visible_answer,
    validate_request,
)
from app.services.response_cache import ResponseCache, build_cache_key


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("LEGALBRAIN_MODEL", "Qwen/Qwen3-4B")

DEFAULT_ADAPTER = "outputs/legalbrain-qwen3-4b-question-only-day2"
CONFIGURED_ADAPTER = os.getenv(
    "LEGALBRAIN_ADAPTER_PATH",
    DEFAULT_ADAPTER,
).strip()

# Use "base" or an empty value in the environment to run without an adapter.
if CONFIGURED_ADAPTER.lower() in {"", "base", "none"}:
    ADAPTER_PATH = ""
else:
    ADAPTER_PATH = str(Path(CONFIGURED_ADAPTER))

model = None
tokenizer = None

response_cache = ResponseCache(max_entries=128, ttl_seconds=300)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=3000)
    max_tokens: int = Field(default=180, ge=32, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    logger.info("Loading model: %s", MODEL_NAME)

    if ADAPTER_PATH:
        if not Path(ADAPTER_PATH).exists():
            raise RuntimeError(
                f"Adapter directory not found: {ADAPTER_PATH}. "
                "Check LEGALBRAIN_ADAPTER_PATH."
            )

        logger.info("Loading LoRA adapter from: %s", ADAPTER_PATH)
        model, tokenizer = load(
            MODEL_NAME,
            adapter_path=ADAPTER_PATH,
        )
    else:
        logger.info("Loading base model without an adapter.")
        model, tokenizer = load(MODEL_NAME)

    logger.info("Model loaded successfully.")

    yield

    model = None
    tokenizer = None
    logger.info("Model resources released.")


app = FastAPI(
    title="LegalBrain AI",
    description="A question-only legal information chatbot using Qwen3-4B.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "app": "LegalBrain AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ready" if model is not None else "not_ready",
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH or "base",
    }


@app.get("/cache/stats")
async def cache_stats() -> dict[str, int]:
    return response_cache.stats()


@app.post("/answer")
async def answer(request: AnswerRequest) -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not ready yet.",
        )

    decision = validate_request(request.question)

    if not decision.allowed:
        raise HTTPException(
            status_code=400,
            detail=decision.reason,
        )

    cache_key = build_cache_key(
        model_name=MODEL_NAME,
        adapter_path=ADAPTER_PATH or "base",
        question=request.question,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        prompt_version=SYSTEM_PROMPT_VERSION,
    )

    start_time = time.perf_counter()

    cached = response_cache.get(cache_key)

    if cached is not None:
        cached["cache_hit"] = True
        cached["latency_ms"] = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
        return cached

    try:
        messages = build_messages(request.question)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=make_sampler(temp=request.temperature),
            verbose=False,
        )

        visible_answer = extract_visible_answer(raw_output)

        if not visible_answer:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The model did not produce a complete visible answer. "
                    "Please try again."
                ),
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed. Check the backend logs.",
        ) from exc

    elapsed_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    result = {
        "question": request.question.strip(),
        "answer": visible_answer,
        "disclaimer": DISCLAIMER,
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH or "base",
        "latency_ms": elapsed_ms,
        "cache_hit": False,
    }

    # Exclude request-specific timing and cache status from the stored payload.
    response_cache.set(cache_key, result)

    return result