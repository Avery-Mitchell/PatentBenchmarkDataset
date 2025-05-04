import json
from sklearn.model_selection import train_test_split
from typing import Optional

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

if __name__ == "__main__":
    data = load_data("temp.json")
    train_data, test_data = split_json(data, test_size=0.2, random_state=100)
    