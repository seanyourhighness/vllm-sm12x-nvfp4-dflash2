#!/usr/bin/env bash
# verify.sh — health + correctness gates for the DFlash2 release.
#
#   ./verify.sh --smoke   # health, model routing, deterministic canary
#   ./verify.sh --full    # smoke + long-decode determinism + NIAH + tools
#   ./verify.sh --vision  # exact two-image fixture and bounded vision gate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f .env ]] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a

base="http://${BIND_ADDRESS}:${VLLM_PORT}"

# --- smoke: health + model routing + deterministic canary -------------------
curl -fsS "$base/health" >/dev/null
response="$(curl -fsS "$base/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 times 23? Reply with only the number.\"}]}")"
python3 - "$response" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
text = (payload["choices"][0]["message"].get("content") or "").strip()
# The DFlash2 release canary: the model must answer 437 exactly.
if "437" not in text:
    raise SystemExit(f"canary failed: expected 437, got {text!r}")
print(f"PASS: health, model routing, and deterministic canary (437)")
PY

if [[ "${1:-}" == "--smoke" ]]; then
  exit 0
elif [[ "${1:-}" == "--vision" ]]; then
  echo "Running exact vision fixture and concurrency gate..."
  BASE_URL="http://${BIND_ADDRESS}:${VISION_PORT}" \
    MODEL="${SERVED_MODEL_NAME}" \
    python3 bench/vision_gate.py
  exit 0
elif [[ "${1:-}" != "--full" && -n "${1:-}" ]]; then
  echo "usage: ./verify.sh [--smoke|--full]" >&2
  exit 2
fi

# --- full: long-decode determinism + NIAH + tools ---------------------------
echo "Running full gate suite..."

# 1) Long-decode determinism: two identical 2048-token decodes must match.
declare -a hashes
for i in 1 2; do
  out="$(curl -fsS "$base/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Write a Python function that implements a thread-safe LRU cache with TTL eviction.\",\"max_tokens\":2048,\"temperature\":0,\"ignore_eos\":true}")"
  h="$(python3 -c 'import json,sys,hashlib; t=json.loads(sys.argv[1])["choices"][0]["text"]; print(hashlib.sha256(t.encode()).hexdigest())' "$out")"
  hashes+=("$h")
done
[[ "${hashes[0]}" == "${hashes[1]}" ]] || { echo "FAIL: long-decode determinism (hashes differ)" >&2; exit 1; }
echo "PASS: long-decode determinism (2/2 identical)"

# 2) NIAH: plant a codeword in a ~30K-token haystack at two depths and
#    require retrieval. Robust to thinking-enabled responses (content may be
#    null while reasoning_content holds the answer) and verifies the server
#    actually ingested the haystack (see bench/niah_gate.py).
BASE_URL="$base" MODEL="${SERVED_MODEL_NAME}" python3 bench/niah_gate.py \
  || { echo "FAIL: NIAH (codeword not retrieved)" >&2; exit 1; }
echo "PASS: NIAH (MOONWEASEL-7 retrieved)"

# 3) Tool calling: the model must emit a valid tool_call.
tool="$(python3 - "$base" "$SERVED_MODEL_NAME" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "temperature": 0,
    "max_tokens": 128,
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": "auto",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req, timeout=120).read().decode())
PY
)"
python3 -c 'import json,sys; p=json.loads(sys.argv[1]); tc=p["choices"][0]["message"].get("tool_calls"); sys.exit(0 if tc and tc[0]["function"]["name"]=="get_weather" else 1)' "$tool" \
  || { echo "FAIL: tool calling (no valid tool_call)" >&2; exit 1; }
echo "PASS: tool calling (get_weather invoked)"

echo "All full gates passed."
