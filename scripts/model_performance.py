import json
from sklearn.model_selection import train_test_split
from typing import Optional
import os
from together import Together
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv("api_key.env")
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Together.ai client
client = Together(api_key=HF_API_KEY) 

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
    
    ]

    # Calculate the performance of each model using the classification report
    print("\n======= Running all models =======")
    for model in models:
        gold_labels = [item["label"] for item in test_data] # Human annotations
        predictions = []

        for item in test_data:
            claim = item["claim"]
            evidence = item["evidence"] 
            
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