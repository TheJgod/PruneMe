import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Path to your pruned Mistral model
model_path = "./merged"

# 4-bit quantization setup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model + tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=quantization_config,
    output_hidden_states=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# In case pad_token is not defined (common for Mistral)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("After Pruning a layer")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {model_size_bytes / (1024 ** 2):.2f} MB")

# ✅ Safe text generation function
def generate_text(input_text):
    # Prepare inputs properly
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # Generate text with attention mask & pad_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        #attention_mask=inputs["attention_mask"],
        do_sample=True,  # enables sampling (optional)
        top_p=0.95,
        temperature=0.8,
    )

    # Decode generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 🔍 Example
input_text = "The future of AI is"
generated_text = generate_text(input_text)
print("\nGenerated text:\n", generated_text)
