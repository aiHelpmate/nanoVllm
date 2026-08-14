"""Test that our Llama implementation produces the same output as HuggingFace."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_vllm.config import ModelConfig
from nano_vllm.model.loader import load_model


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_forward_pass_matches_hf():
    """Test that our model's forward pass matches HuggingFace's output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use float32 for precise comparison

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    hf_model.eval()

    # Load our model
    print("Loading nano-vllm model...")
    our_model = load_model(MODEL_ID, device=device, dtype=dtype)

    # Tokenize a test input
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    test_input = "The capital of France is"
    input_ids = tokenizer.encode(test_input, return_tensors="pt").to(device)

    print(f"Input: {test_input}")
    print(f"Input IDs: {input_ids}")

    # Get HuggingFace output
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

    # Get our output (without KV cache for fair comparison)
    with torch.no_grad():
        our_logits = our_model(input_ids)

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Our logits shape: {our_logits.shape}")

    # Compare
    max_diff = (hf_logits - our_logits).abs().max().item()
    mean_diff = (hf_logits - our_logits).abs().mean().item()

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    # Check if predictions match
    hf_pred = hf_logits[:, -1, :].argmax(dim=-1)
    our_pred = our_logits[:, -1, :].argmax(dim=-1)

    print(f"HF predicted token: {hf_pred.item()} ({tokenizer.decode(hf_pred)})")
    print(f"Our predicted token: {our_pred.item()} ({tokenizer.decode(our_pred)})")

    # Assertions
    assert hf_logits.shape == our_logits.shape, "Shape mismatch!"
    assert max_diff < 1e-3, f"Logits differ too much: max_diff={max_diff}"
    assert hf_pred.item() == our_pred.item(), "Predicted tokens don't match!"

    print("\nAll tests passed!")


def test_generation_matches_hf():
    """Test that greedy generation matches HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # Load models
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Generate with HuggingFace
    test_input = "The capital of France is"
    input_ids = tokenizer.encode(test_input, return_tensors="pt").to(device)

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    print(f"HuggingFace output: {hf_text}")

    # Generate with our engine
    from nano_vllm.engine import LLMEngine

    engine = LLMEngine(MODEL_ID, device=device, dtype=dtype)
    our_text = engine.generate(test_input, max_tokens=20)
    print(f"nano-vllm output: {our_text}")

    # Compare (they should be identical for greedy decoding)
    assert hf_text == our_text, f"Outputs differ!\nHF: {hf_text}\nOurs: {our_text}"
    print("\nGeneration test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    test_forward_pass_matches_hf()

    print("\n" + "=" * 60)
    print("Testing generation...")
    print("=" * 60)
    test_generation_matches_hf()