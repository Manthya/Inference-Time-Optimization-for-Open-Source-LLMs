#!/usr/bin/env python3
"""
Test script to verify Qwen greedy decoding works correctly.
Tests the proposed approach with a simple test query.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen_greedy_decoding():
    """Test Qwen model with greedy decoding on sample queries."""
    
    print("="*80)
    print("Testing Qwen Greedy Decoding")
    print("="*80)
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Use a simple test prompt (GSM8K-style math problem)
    print("\n" + "="*80)
    print("Running Greedy Decoding Test")
    print("="*80)
    
    # Test prompt
    prompt = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
    
    print(f"\nPrompt:\n{prompt}")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(f"\nChat template applied (length: {len(text)} chars)")
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # GREEDY DECODING
    print("\nGenerating with greedy decoding...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding
            num_beams=1,      # Ensure greedy (not beam search)
        )
    
    # Decode
    input_length = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Show results
    print(f"\nGenerated {len(output_ids)} tokens")
    print(f"\nGenerated Output:")
    print("-"*80)
    print(response)
    print("-"*80)
    
    expected_answer = "72"
    print(f"\nExpected answer: {expected_answer}")
    
    # Check for issues
    if len(output_ids) == 0:
        print("\n⚠️  WARNING: Generated 0 tokens!")
    elif len(output_ids) == 512:
        print("\n⚠️  WARNING: Generated max tokens (512) - may not have stopped at EOS")
    else:
        print(f"\n✓ Generated {len(output_ids)} tokens (stopped naturally)")
    
    # Check if output is all same token (bug indicator)
    unique_tokens = len(set(output_ids.tolist()))
    if unique_tokens == 1:
        print(f"\n⚠️  WARNING: All tokens are the same (token ID: {output_ids[0].item()})!")
        print("This indicates the token 0 bug is present!")
    else:
        print(f"\n✓ Output has {unique_tokens} unique tokens")
    
    # Show first 10 token IDs for debugging
    print(f"\nFirst 10 generated token IDs: {output_ids[:10].tolist()}")
    print(f"Last 10 generated token IDs: {output_ids[-10:].tolist()}")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nSummary:")
    print(f"- Model: {model_name}")
    print(f"- Method: Greedy decoding (do_sample=False)")
    print(f"- Generated: {len(output_ids)} tokens")
    print(f"- Unique tokens: {unique_tokens}")
    print("\nIf you see proper text output above (not all same token),")
    print("then this approach works and should be integrated into stage_0_baseline.py")


if __name__ == "__main__":
    try:
        test_qwen_greedy_decoding()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
