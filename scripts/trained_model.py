import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "dataset.json" 
from api_keys import HUGGINGFACE_API_KEY


PROMPT_TEMPLATE: str = (
    "<s>[INST] You are an expert in evaluating patent-related claims. "
    "Given the following claim and supporting evidence, determine whether the claim is true, false, or lacks enough information.\n"
    "Claim: {claim}\n"
    "Evidence: {evidence}\n"
    "Answer with 'True', 'False', 'Mixture', or 'Not Enough Information' and explain your reasoning. [/INST]"
)

def load_dataset(dataset_path: str) -> Dataset:
    """
    Loads the dataset from a JSON file and formats it for training.

    Arguments:
        dataset_path: path to the JSON file

    Returns:
        Hugging face dataset object {prompt, output}
    """

    # Open the JSON file and load the data
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Samples to be returned - used for training
    samples = []
    for item in raw:
        claim = item.get("claim", "")
        evidence = item.get("evidence", [])
        evidence_str = "\n".join(evidence) if isinstance(evidence, list) else str(evidence)
        prompt = PROMPT_TEMPLATE.format(claim=claim, evidence=evidence_str)
        label = item.get("label", "Not Enough Information")
        samples.append({"prompt": prompt, "label": label})

    dataset = Dataset.from_list(samples)
    print(f"Loaded {len(dataset)} examples.")
    return dataset

def tokenize(example: dict[str, str], tokenizer: PreTrainedTokenizerBase) -> dict[str, any]:
    """
    Tokenizes the input examples.

    Arguments:
        example: a dictionary containing the input text

    Returns:
        tokenized input IDs and attention masks
    """
    # Text to be tokenized
    full_text: str = example["prompt"]

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
    Main function to train the model using LoRA.
    """
    dataset: Dataset = load_dataset(DATASET_PATH)
    print(f"\nLoaded dataset with {len(dataset)} samples.\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=True, 
        token=HUGGINGFACE_API_KEY,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=False)
    tokenized_dataset = tokenized_dataset.remove_columns(["label"])
    print("\nTokenization complete.\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./lora-patent-misinformation-model",
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=4,  
        num_train_epochs=20, 
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=1e-4,
        fp16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting training...\n")
    trainer.train()
    print("\nTraining complete.\n")
    print("Saving LoRA model...\n")

    model.save_pretrained("./lora-patent-misinformation-model")

if __name__ == "__main__":
    train_model()