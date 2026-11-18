"""The backbone, held fixed across every condition.

A1-A4 differ only in control flow and in what enters the context window, so the
model, the decoding settings and the token accounting all live here and nowhere
else.  Every call is counted: token cost is one of the axes the comparison is
scored on, and an agent that wins on success while spending three times the
tokens has not obviously won.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached_tokens: int = 0      # prompt tokens the provider billed at its cached rate (OpenAI)
    malformed: int = 0          # responses that were not usable JSON
    repairs: int = 0            # retries that then parsed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "llm_seconds": round(self.seconds, 2),
            "malformed_json": self.malformed,
            "json_repairs": self.repairs,
        }


@dataclass
class LLMBackbone:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 256
    seed: int = 0
    timeout: float = 300.0
    usage: Usage = field(default_factory=Usage)
    trace: List[Dict[str, Any]] = field(default_factory=list)   # one entry per call: what came back
    keep_messages: bool = False                                  # also keep the full prompt of each call
    # Ollama is not bit-reproducible at temperature 0: the same prompt can come back
    # with a different answer on another run.  With a cache file, each distinct
    # prompt is put to the model once and its answer reused, so two conditions
    # behave differently only from the point where their prompts differ.
    cache_path: Optional[str] = None
    _cache: Optional[Dict[str, Any]] = None
    # Models that can "think" before they answer are run with thinking off (False), so every
    # model answers within the same num_predict; None leaves the request as it always was.
    think: Optional[bool] = None
    last_error: str = ""
    # A model named "openai:<id>" is asked through the OpenAI API (key in OPENAI_API_KEY);
    # `effort` is its reasoning effort, "none" being the counterpart of think=False.
    effort: str = "none"

    def _load_cache(self) -> Dict[str, Any]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    for line in f:
                        if line.strip():
                            e = json.loads(line)
                            self._cache[e["key"]] = e
        return self._cache

    # ------------------------------------------------------------------ plumbing
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.host.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _openai(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """One call to the OpenAI chat API, returned in the shape of an Ollama reply."""
        fmt = ({"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": False}}
               if schema else {"type": "json_object"})
        body = {"model": self.model.split(":", 1)[1], "messages": messages, "response_format": fmt,
                "max_completion_tokens": self.num_predict, "seed": self.seed}
        if self.effort:
            body["reasoning_effort"] = self.effort
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        for wait in (2, 8, 30, None):                       # rate limits and server errors are retried
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or wait is None:
                    raise urllib.error.URLError(f"OpenAI {e.code}: {e.read()[:300]!r}")
                time.sleep(wait)
        use = out.get("usage") or {}
        return {"message": {"content": out["choices"][0]["message"].get("content") or ""},
                "prompt_eval_count": use.get("prompt_tokens", 0), "eval_count": use.get("completion_tokens", 0),
                "cached_tokens": (use.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

    def chat_json(self, messages: List[Dict[str, str]],
                  retries: int = 2, schema: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """One constrained-decoding call.  Returns a parsed object, or None if
        the model could not be made to emit JSON -- which is itself a datum, so
        it is counted rather than raised."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # a JSON schema (when given) fixes field order as well as shape, so a
            # "thought" field listed first is actually written before the choice
            "format": schema or "json",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        if self.think is not None:
            body["think"] = self.think
        remote = self.model.startswith("openai:")
        key = None
        if self.cache_path:
            ident = [self.model, self.temperature, self.seed, self.num_ctx, self.num_predict, schema, messages]
            if self.think is not None or remote:            # earlier cache files keep their keys
                ident.append({"think": self.think, "effort": self.effort if remote else None})
            key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
            hit = self._load_cache().get(key)
            if hit is not None:
                self.usage.calls += 1
                self.usage.prompt_tokens += hit["pt"]
                self.usage.output_tokens += hit["ot"]
                self.usage.seconds += hit["sec"]
                self.usage.cached_tokens += hit.get("ct", 0)
                entry = {"out": hit["out"]}
                if self.keep_messages:
                    entry["messages"] = messages
                self.trace.append(entry)
                return hit["out"]
        attempt_msgs = list(messages)
        for attempt in range(retries + 1):
            body["messages"] = attempt_msgs
            t0 = time.time()
            try:
                r = self._openai(attempt_msgs, schema) if remote else self._post("/api/chat", body)
            except (urllib.error.URLError, TimeoutError) as e:
                self.last_error = str(e)
                self.usage.calls += 1
                self.usage.seconds += time.time() - t0
                self.usage.malformed += 1
                if attempt == retries:
                    return None
                continue
            self.usage.calls += 1
            self.usage.seconds += time.time() - t0
            self.usage.prompt_tokens += int(r.get("prompt_eval_count", 0))
            self.usage.output_tokens += int(r.get("eval_count", 0))
            self.usage.cached_tokens += int(r.get("cached_tokens", 0))
            text = (r.get("message") or {}).get("content", "")
            try:
                obj = json.loads(text)
                if attempt > 0:
                    self.usage.repairs += 1
                entry: Dict[str, Any] = {"out": obj}
                if self.keep_messages:
                    entry["messages"] = attempt_msgs
                self.trace.append(entry)
                if key is not None and attempt == 0:
                    rec = {"key": key, "out": obj, "pt": int(r.get("prompt_eval_count", 0)),
                           "ot": int(r.get("eval_count", 0)), "sec": round(time.time() - t0, 3)}
                    if r.get("cached_tokens"):
                        rec["ct"] = int(r["cached_tokens"])
                    self._cache[key] = rec
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                return obj if isinstance(obj, dict) else {"value": obj}
            except json.JSONDecodeError:
                self.usage.malformed += 1
                if attempt == retries:
                    return None
                attempt_msgs = list(messages) + [
                    {"role": "assistant", "content": text[:500]},
                    {"role": "user", "content": "That was not valid JSON. Reply with one JSON object only."},
                ]
        return None

    def reset_usage(self) -> None:
        self.usage = Usage()
        self.trace = []
