import os
import torch
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, BitsAndBytesConfig
from peft import PeftModel

# Load API key from .env file
load_dotenv("api_key.env")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

if HF_TOKEN is None:
    raise ValueError("HUGGINGFACE_API_KEY not found in api_key.env")

# Model Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_PATH = "./lora-patent-misinformation-model"
USE_4BIT = True

# Prompt Template
PROMPT_TEMPLATE = (
    "<s>[INST] You are an expert in evaluating patent-related claims. Before you answer, consider the context and the evidence provided."
    "Given the following claim and supporting evidence, determine whether the claim is true, false, or lacks enough information.\n"
    "Claim: {claim}\n"
    "Evidence: {evidence}\n"
    "Answer only with 'True', 'False', 'Mixture', or 'Not Enough Information'[/INST]"
)

def evaluate_claim(model: str, tokenizer: PreTrainedTokenizerBase, claim: str, evidence: list[str]) -> str:
    """
    Completes the claim evaluation using the fine-tuned model.

    Arguments:
        model: the fine-tuned model
        tokenizer: the tokenizer for the model
        claim: the claim to be evaluated
        evidence: a list of evidence strings

    Returns:
        The model's response to the claim evaluation
    """
    evidence_text = "\n".join(evidence)
    prompt = PROMPT_TEMPLATE.format(claim=claim, evidence=evidence_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.01,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

def normalize_response(response: str) -> str:
    """
    Normalize the response from the LLM API

    Arguments:
        response: the response from the LLM API

    Returns:
        Normalized response
    """

    if "True" in response:
        return "True"
    elif "False" in response:
        return "False"
    else:
        return "Not Enough Information"

def main():
    """
    Main function to run the command line interface for evaluating patent claims.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Patent Claim Evaluator CLI")
    parser.add_argument("--claim", type=str, required=True, help="The patent claim to evaluate")
    parser.add_argument("--evidence", nargs="+", required=True, help="Supporting evidence sentences")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)
    model.eval()

    # Evaluate and print result
    result = normalize_response(evaluate_claim(model, tokenizer, args.claim, args.evidence))
    print("\n=== Model Response ===\n")
    print(result)

if __name__ == "__main__":
    main()