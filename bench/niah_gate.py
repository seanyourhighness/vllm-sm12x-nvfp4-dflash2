#!/usr/bin/env python3
"""NIAH (needle-in-a-haystack) retrieval gate for the DFlash2 release.

The old inline NIAH in verify.sh had two defects:
  1. The haystack was built but NEVER sent — the request only contained the
     question, so retrieval was impossible by construction (the gate could
     never meaningfully pass).
  2. The check assumed `message.content` is a string. With thinking enabled
     (this chat template defaults enable_thinking=true, effort=xhigh), vLLM
     returns `content: null` and puts the reasoning in `reasoning_content`,
     so `"MOONWEASEL-7" in None` crashed with
     `TypeError: argument of type 'NoneType' is not a container or iterable`.

This gate fixes both:
  * Builds a real, deterministic haystack and plants the codeword at one or
    more token depths (default two depths).
  * Verifies the server actually ingested the haystack
    (usage.prompt_tokens must cover it) — structurally impossible to pass
    without the needle being in context.
  * Extracts the answer from `content` OR `reasoning_content`, so a
    thinking-enabled response can still pass without crashing.
  * Disables thinking for the request (like bench/vision_gate.py) so the
    codeword lands in `content` when the server honors the template kwarg,
    and the reasoning_content fallback covers servers that ignore it.

Exit 0 on PASS, 1 on FAIL.
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18089")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-nvfp4-dflash2")
CODEWORD = "MOONWEASEL-7"
# ~30K-token haystack (well under the 262144 MAX_MODEL_LEN). Filler is ~4
# chars/token, so 30000 tokens ~= 120000 chars. Env-overridable.
HAYSTACK_TOKENS = int(os.environ.get("NIAH_HAYSTACK_TOKENS", "30000"))
# Needle positions as fractions of the haystack (env-overridable).
DEPTHS = [float(x) for x in os.environ.get("NIAH_DEPTHS", "0.25,0.75").split(",")]

_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz. "
    "The jay, pig, fox, zebra and my wolves quack. "
    "Waltz, bad nymph, for quick jigs vex. "
    "Glib jocks quiz nymphs to vex dwarf. "
    "Squdgy fez, blank jimp crwth vox. "
)
_CHARS_PER_TOKEN = 4.0


def haystack_for(depth):
    """Deterministic ~N-token filler with the codeword planted at `depth`."""
    target_chars = int(HAYSTACK_TOKENS * _CHARS_PER_TOKEN)
    needle_pos = int(target_chars * depth)
    block = _FILLER * (target_chars // len(_FILLER) + 1)
    left, right = block[:needle_pos], block[needle_pos:]
    return left + f"\n\nSECRET_CODEWORD={CODEWORD}\n\n" + right


def post(payload, timeout=900):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.monotonic() - t


def retrieve(depth):
    """One NIAH probe: plant at `depth`, ask, return (text, usage, elapsed)."""
    haystack = haystack_for(depth)
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": haystack + "\n\nQuestion: What is the SECRET_CODEWORD "
                                   "value? Reply with only the codeword.",
        }],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data, elapsed = post(payload)
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
    usage = data.get("usage", {})
    return text, usage, elapsed


def main():
    if not 0.0 < min(DEPTHS) or max(DEPTHS) >= 1.0:
        print(f"FAIL: NIAH_DEPTHS must be in (0,1), got {DEPTHS}", file=sys.stderr)
        return 1
    failures = []
    for depth in DEPTHS:
        try:
            text, usage, elapsed = retrieve(depth)
        except Exception as e:  # network / malformed response
            failures.append({"depth": depth, "error": f"{type(e).__name__}: {e}"})
            print(f"FAIL: NIAH depth={depth:.0%} -> request error: {e}", file=sys.stderr)
            continue
        prompt_tok = usage.get("prompt_tokens") or 0
        # The server must have actually ingested the haystack (~30K tokens).
        ingested = prompt_tok >= int(HAYSTACK_TOKENS * 0.5)
        found = CODEWORD in text
        print(json.dumps({
            "depth": round(depth, 3),
            "retrieved": found,
            "prompt_tokens": prompt_tok,
            "haystack_tokens_ingested": ingested,
            "completion_tokens": usage.get("completion_tokens"),
            "elapsed_s": round(elapsed, 2),
            "response": text.strip()[:200],
        }, indent=2))
        if not ingested:
            failures.append({"depth": depth, "reason": "haystack not ingested"})
            print(f"FAIL: NIAH depth={depth:.0%} -> haystack not ingested "
                  f"(prompt_tokens={prompt_tok})", file=sys.stderr)
        elif not found:
            failures.append({"depth": depth, "reason": "codeword not retrieved"})
            print(f"FAIL: NIAH depth={depth:.0%} -> codeword not retrieved",
                  file=sys.stderr)
    if failures:
        return 1
    print(f"PASS: NIAH ({CODEWORD} retrieved at {len(DEPTHS)} depth(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
