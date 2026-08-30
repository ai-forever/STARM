"""FastAPI app exposing a loaded checkpoint.

One generation endpoint, ``POST /generate``, which takes either a single prompt object or a
list of them and mirrors that shape in its response. A list is batched, which never changes an
answer: each prompt gets the same output and the same ``steps`` it would get on its own.

The body is parsed by hand rather than declared as a Pydantic parameter, so a request without
``content-type: application/json`` is read the same as one with it — ``curl -d '{...}'`` works
as typed. The handler is therefore ``async def``; the model work goes to a threadpool so a slow
request never blocks the event loop, and ``StarmEngine`` serialises it on its own lock anyway.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool

from .engine import Answer, Prompt, StarmEngine
from .tokenizers import TokenizerError

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """One prompt. Post a single object, or a list of them for a batch."""

    task: str = Field(
        ..., description="The task this prompt is for; must be the one this server serves"
    )
    input: str = Field(..., description="The prompt, in this task's text format; see GET /info")
    puzzle_id: Optional[str] = Field(
        None,
        description=(
            "ARC only, and required there: the task's id as ARC-AGI names it, e.g. "
            "'007bbfb7'. It selects the learned puzzle embedding, which is where a "
            "checkpoint keeps what it knows about that task. Checked against the "
            "identifiers.json in the checkpoint directory."
        ),
    )

    def to_prompt(self) -> Prompt:
        return Prompt(input=self.input, puzzle_id=self.puzzle_id)


class GenerateResponse(BaseModel):
    task: str = Field(..., description="Echoed from the request")
    input: str = Field(..., description="Echoed from the request")
    output: str = Field(..., description="The model's answer, in this task's text format")
    steps: int = Field(
        ..., description="Recursion steps this example took before its ACT head said stop"
    )
    max_steps: int = Field(..., description="halt_max_steps from the checkpoint's config")
    puzzle_id: Optional[str] = Field(None, description="Echoed when the request carried one")

    @classmethod
    def from_answer(cls, request: GenerateRequest, answer: Answer) -> "GenerateResponse":
        return cls(
            task=request.task,
            input=request.input,
            output=answer.output,
            steps=answer.steps,
            max_steps=answer.max_steps,
            puzzle_id=request.puzzle_id,
        )


_REQUESTS = TypeAdapter(Union[GenerateRequest, List[GenerateRequest]])


def create_app(engine: StarmEngine) -> FastAPI:
    info = engine.info()
    served_task = info["task"]
    app = FastAPI(
        title="STARM inference API",
        description=(
            f"Serving a {info['arch']} checkpoint trained on the '{served_task}' task.\n\n"
            f"**Input format:** {info['input_format']}\n\n"
            "POST /generate takes one prompt object or a list of them; the response mirrors "
            "the shape you sent, and batching never changes an answer."
        ),
        version="1.0.0",
    )

    @app.exception_handler(TokenizerError)
    def _on_tokenizer_error(_request: Request, exc: TokenizerError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", summary="Liveness probe")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", summary="Checkpoint and task metadata")
    def get_info() -> Dict[str, Any]:
        return engine.info()

    def _check(request: GenerateRequest) -> None:
        """One checkpoint serves one task, so a mismatch is a misrouted client."""
        if request.task != served_task:
            raise HTTPException(
                status_code=422,
                detail=f"this server serves the {served_task!r} task, not {request.task!r}",
            )
        if request.puzzle_id is not None and served_task != "arc":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"puzzle_id is an ARC field; the {served_task!r} task trains a single "
                    "blank puzzle identifier"
                ),
            )

    @app.post(
        "/generate",
        response_model=Union[GenerateResponse, List[GenerateResponse]],
        summary="Answer one prompt, or a batch of them",
    )
    async def generate(
        http_request: Request,
    ) -> Union[GenerateResponse, List[GenerateResponse]]:
        body = await http_request.body()
        try:
            payload = _REQUESTS.validate_python(json.loads(body))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"body is not JSON: {exc}") from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc

        single = isinstance(payload, GenerateRequest)
        requests: List[GenerateRequest] = [payload] if single else payload
        if not requests:
            raise HTTPException(status_code=400, detail="send at least one prompt")
        for request in requests:
            _check(request)

        try:
            answers = await run_in_threadpool(
                engine.generate, [r.to_prompt() for r in requests]
            )
        except TokenizerError:
            raise  # handled above -> 422
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        responses = [
            GenerateResponse.from_answer(request, answer)
            for request, answer in zip(requests, answers)
        ]
        return responses[0] if single else responses

    return app
