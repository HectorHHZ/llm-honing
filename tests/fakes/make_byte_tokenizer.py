"""Writes ``tests/fakes/byte_tokenizer/``: a Hugging Face tokenizer whose token ids are the UTF-8 bytes of the text.

Slime's adapter needs a tokenizer with a chat template; this one loads offline with
``AutoTokenizer``, and its ids are byte values, so ``tests/fakes/sglang_generate.py``
decodes a prompt with ``bytes(ids)`` and needs no ``transformers``. Run it with a
Python that has ``tokenizers`` and ``transformers`` (the slime agent environment from
``scripts/setup-slime-agent-env.sh``):

    python tests/fakes/make_byte_tokenizer.py tests/fakes/byte_tokenizer

The chat template prints each message as ``<|role|>`` and its content, a tool call as
``<call NAME>``, and the tool names on a first ``<|tools|>`` line.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TEMPLATE = (
    "{% if tools %}<|tools|>{% for t in tools %} {{ t['function']['name'] }}{% endfor %}\n{% endif %}"
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}"
    "{% if m.get('tool_calls') %}{% for c in m['tool_calls'] %}"
    "<call {{ c['function']['name'] }}>{% endfor %}{% endif %}"
    "\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-character table, which the ByteLevel pre-tokenizer uses."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), 256))
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, characters), strict=True))


def main(target: str) -> None:
    vocab = {character: byte for byte, character in bytes_to_unicode().items()}
    vocab["<|eos|>"] = 256
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|eos|>")
    fast.chat_template = TEMPLATE
    fast.save_pretrained(target)
    for path in Path(target).iterdir():  # end each file with a newline, as the repository's hooks do
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            path.write_text(text + "\n", encoding="utf-8")
    loaded = AutoTokenizer.from_pretrained(target)
    text = "héllo <tool_call>\n{ä} ✓"
    ids = loaded.encode(text, add_special_tokens=False)
    assert ids == list(text.encode("utf-8")), "token ids must be the text's bytes"
    assert loaded.decode(ids) == text


if __name__ == "__main__":
    main(sys.argv[1])
