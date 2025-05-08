import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from typing import Optional
from together import Together

from api_keys import TOGETHER_API_KEY

# Together.ai client
client = Together(api_key=TOGETHER_API_KEY) # Uses TOGETHER_API_KEY from environment variables, otherwise set manually

# === DO NOT UPLOAD API KEYS TO GITHUB ===

def load_data(path: str) -> list[dict[str, any]]:
    """
    Loads the JSON dataset in

    Arguments:
        path: path to the JSON file
    
    Returns:
        List of dictionaries representing the dataset
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at top level in {path}")
    return data

def split_json(
    items: list[dict[str, any]],
    test_size: float = 0.2,
    random_state: int = 100
) -> tuple[list[dict[str, any]], list[dict[str, any]]]:
    """
    Split a list of JSON-like dicts into train/test lists.

    Arguments:
        items: list of dictionaries (json instances) to split
        test_size: proportion of the dataset to include in the test split
        random_state: random seed for reproducibility

    Returns:
        Tuple of two lists: (train_items, test_items)
    """
    # Extract labels for stratification
    labels = [item.get("label") for item in items]
    unique_labels = set(labels)
    stratify: Optional[list[str]] = labels if len(unique_labels) > 1 else None

    train_items, test_items = train_test_split(
        items,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_items, test_items

def query_together_api(claim: str, evidence: str, model: str) -> str:
    """
    Query the Together.api with claim and evidence

    Arguments:
        claim: the claim to be verified
        evidence: the evidence to be used for verification
        model: which model to use from Together.ai 

    Returns:
        The response from LLM API
    """

    # Prompt for LLM
    prompt = f"""
Claim: {claim}
Evidence: {evidence}
Is the claim true based on the evidence? Answer with either "True", "False", or "Not Enough Information".
    """

    # Make the API request
    api_request = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )
    return api_request.choices[0].message.content.strip()

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

def run_all_models(claim: str, evidence: str):
    models: list[str] = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",


    ]
    
    return 0

def evaluate_model():
    return 0

if __name__ == "__main__":
    # Load the in the data from the JSON file
    data = load_data("temp.json")

    # Split the data into training and testing sets
    # Use random_state=100 for reproducibility -> delete for non-deterministic splits
    #train_data, test_data = split_json(data, test_size=0.2, random_state=100)

    claim: str = "The sky is red."
    evidence: str = "When I went outside, the sky was red."
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    response: str = query_together_api(claim=claim, evidence=evidence, model=model)
    prediction: str = normalize_response(response)
    
    print(f"Claim: {claim}")
    print(f"Evidence: {evidence}")
    print(f"Prediction: {prediction}")