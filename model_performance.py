import json
from sklearn.model_selection import train_test_split
from typing import Optional
from together import Together
from sklearn.metrics import classification_report

#TODO: Modify dataset.json so that only True/False/Not Enough Information are used as labels
#TODO: Look at the different models available in Together.ai - change models in run_all_models() to use the best ones
#TODO: Add variable for end users to add their own API key

from api_keys import TOGETHER_API_KEY

# Together.ai client
client = Together(api_key=TOGETHER_API_KEY) 

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

# This is unused now, but keeping it here for reference
"""
def split_json(
    items: list[dict[str, any]],
    test_size: float = 0.2,
    random_state: int = 100
) -> tuple[list[dict[str, any]], list[dict[str, any]]]:
    \"""
    Split a list of JSON-like dicts into train/test lists.

    Arguments:
        items: list of dictionaries (json instances) to split
        test_size: proportion of the dataset to include in the test split
        random_state: random seed for reproducibility

    Returns:
        Tuple of two lists: (train_items, test_items)
    \"""

    # Filter out invalid entries
    valid_items = [item for item in items if item["claim"] != "N/A" and item["label"] != "N/A"]

    labels = [item["label"] for item in valid_items]
    stratify = labels if len(set(labels)) > 1 else None

    train_items, test_items = train_test_split(
        valid_items,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_items, test_items
"""
    
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

def run_all_models(test_data: list[dict[str, any]]) -> None:

    models = [                
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    #"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    #"meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #"Qwen/Qwen2.5-Coder-32B-Instruct",
    #"Qwen/Qwen2-72B-Instruct",
    #"arcee_ai/arcee-spotlight",
    #"deepseek-ai/DeepSeek-V3",
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    #"google/gemma-2b-it"
    ]

    # Calculate the performance of each model using the classification report
    print("\n======= Running all models =======")
    for model in models:
        gold_labels = [item["label"] for item in test_data] # Human annotations
        predictions = []

        for item in test_data:
            claim = item["claim"]
            evidence = item["context"]["text"] # Change this once the evidence is added to the JSON - just passing the whole article in the meantime
            
            try:
                response = query_together_api(claim=claim, evidence=evidence, model=model)
                prediction = normalize_response(response)
            except Exception as e:
                print(f"Error querying model {model}")
                prediction = "Error in model query"
            
            predictions.append(prediction)

        print(f"\nModel: {model}")
        print(classification_report(gold_labels, predictions, digits=3))
        print("\n")
    

if __name__ == "__main__":
    # Load the in the data from the JSON file
    data = load_data("dataset.json")

    run_all_models(data)