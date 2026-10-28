import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(model_name=MODEL_NAME, device=None):
    if device is None:
        device = choose_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.float16
        if device.type in ("mps", "cuda")
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device