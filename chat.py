"""Chat directly with the USB Qwen GGUF model; no Ollama server needed."""
import argparse
import json
from pathlib import Path
import re
import sys


def find_model(directory):
    manifest = directory / "manifests/registry.ollama.ai/library/qwen3/8b-q4_K_M"
    data = json.loads(manifest.read_text())
    layer = next(x for x in data["layers"]
                 if x["mediaType"] == "application/vnd.ollama.image.model")
    digest = layer["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid model digest in manifest")
    path = directory / "blobs" / digest.replace(":", "-")
    if path.stat().st_size != layer["size"]:
        raise ValueError("Model file is incomplete")
    return path


def chat(model, temperature, max_tokens):
    messages = []
    print("Ready. Type a message; /new starts fresh; /quit exits.\n"
          "Conversation is kept in memory only. Ctrl+C cancels a reply.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            return
        if prompt in {"/new", "/clear"}:
            messages.clear()
            model.reset()
            print("New conversation started.\n")
            continue
        if prompt == "/help":
            print("/new: forget conversation | /quit: exit | /help: commands\n")
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        answer = ""
        stream = None
        try:
            print("Model: ", end="", flush=True)
            stream = model.create_chat_completion(
                messages=candidate, temperature=temperature,
                max_tokens=max_tokens, stream=True,
            )
            reason = None
            for chunk in stream:
                choice = chunk["choices"][0]
                text = choice.get("delta", {}).get("content") or ""
                answer += text
                print(text, end="", flush=True)
                reason = choice.get("finish_reason") or reason
            messages = candidate + [{"role": "assistant", "content": answer}]
            print("\n")
            if reason == "length":
                print("[Reply reached the token limit. Ask to continue, or use /new.]\n")
        except KeyboardInterrupt:
            print("\n[Cancelled. This exchange was not added to the conversation.]\n")
        except Exception as exc:
            print(f"\n[Could not generate reply: {exc}]\n"
                  "If the conversation is too long, use /new.\n", file=sys.stderr)
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "OllamaModels")
    parser.add_argument("--model-file", type=Path, help="Use a different local GGUF file")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU offloading")
    args = parser.parse_args()
    if args.context < 512 or not 0 < args.max_tokens < args.context:
        parser.error("Use context >= 512 and 0 < max-tokens < context")
    if not 0 <= args.temperature <= 2:
        parser.error("Temperature must be between 0 and 2")
    try:
        path = args.model_file or find_model(args.models_dir)
        with path.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("Selected model is not a GGUF file")
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Cannot read USB model: {exc}\nUse --models-dir or --model-file to select it.\n")
    try:
        from llama_cpp import Llama
    except ImportError:
        parser.exit(1, "Install the local runner first:\n"
                    '.venv/bin/python -m pip install llama-cpp-python -C cmake.args=-DGGML_METAL=on\n')
    print(f"Loading model from {path}\nThe first load from USB may take a while.", flush=True)
    try:
        model = Llama(model_path=str(path), n_ctx=args.context,
                      n_gpu_layers=0 if args.cpu else -1, verbose=False)
    except Exception as exc:
        parser.exit(1, f"Could not load model: {exc}\n")
    try:
        chat(model, args.temperature, args.max_tokens)
    finally:
        model.close()


if __name__ == "__main__":
    main()
