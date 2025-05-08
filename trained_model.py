import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import Any

MODEL_NAME = "name of model to use"
DATASET_PATH = "path to dataset"

def load_dataset(dataset_path: str) -> Dataset:
    """
    Loads the dataset from a JSON file and formats it for training.

    Arguments:
        dataset_path: path to the JSON file

    Returns:
        Hugging face dataset object {prompt, output}
    """

    # Open the JSON file and load the data
    with open(dataset_path, "r") as f:
        raw = json.load(f)

    # Samples to be returned - used for training
    samples = []
    for item in raw:
        prompt = f"Claim: {item["claim"]}\nEvidence: {item["evidence"]}\nIs the claim true based on the evidence? Answer with either 'True', 'False', or 'Not Enough Information'."
        samples.append({"prompt": prompt, "output": item["label"]})

    return Dataset.from_list(samples)

def tokenize(example: dict[str, str], tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """
    Tokenizes the input examples.

    Arguments:
        example: a dictionary containing the input text

    Returns:
        tokenized input IDs and attention masks
    """
    # Text to be tokenized
    full_text: str = f"{example['prompt']} {example['output']}"

    # Tokenize the text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def train_model() -> None:
    """
    Main training loop for LoRA

    Arguments:
        None
    
    Returns:
        None
    """

    # Load the dataset
    dataset: Dataset = load_dataset(DATASET_PATH)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token # Make sure that model has a padding token and that tokenizer does not have a pad token

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=False
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,          # use 4-bit quantization for VRAM efficiency
        device_map="auto",          # automatically place model on GPU
        torch_dtype=torch.float16,  # precision
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,           # LoRA for causal language modeling
        r=8,                                    # trainable parameters
        lora_alpha=16,                          # scaling factor
        lora_dropout=0.05,                      # regularization
        bias="none",                            # no bias
        target_modules=["q_proj", "v_proj"],    # query and value projections
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora-patent-misinformation-model",    # output directory
        per_device_train_batch_size=4,                      # batch size  
        num_train_epochs=10,                                # number of passes over the dataset
        logging_steps=10,                                   # how often to log training steps
        save_strategy="no",                                 # does not save model checkpoints 
        learning_rate=2e-4,                                 # starting learning rate
        fp16=True                                           # use 16-bit floating point precision - reduces VRAM usage
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained("./lora-patent-misinformation-model")

if __name__ == "__main__":
    train_model()