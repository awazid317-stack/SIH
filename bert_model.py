import logging

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
NUM_LABELS = 5
OUTPUT_DIR = "./legal_bert_final"

data = {
    "query": [
        "Mere boss ne 3 mahine se salary nahi di",
        "Police ne bina warrant ke ghar search kiya",
        "Padosi ne meri zameen par kabza kar liya",
        "Patni alag rehna chahti hai",
        "Contract todne par company ne damages nahi diye",
    ]
    * 100,
    "label": [0, 1, 3, 4, 2] * 100,
}

df = pd.DataFrame(data)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_dataset = Dataset.from_pandas(
    train_df[["query", "label"]],
    preserve_index=False,
)
test_dataset = Dataset.from_pandas(
    test_df[["query", "label"]],
    preserve_index=False,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["query"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["query"])
test_dataset = test_dataset.remove_columns(["query"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

results = trainer.evaluate()
print(f"Evaluation Results: {results}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved!")

label_map = {
    0: "Labor",
    1: "Criminal",
    2: "Civil",
    3: "Property",
    4: "Family",
}


def predict(query_text):
    model.eval()

    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    pred = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, pred].item()

    return label_map[pred], confidence


test_query = "Mere boss ne 3 mahine se salary nahi di"
category, confidence = predict(test_query)

print(f"\nQuery: {test_query}")
print(f"Predicted: {category} ({confidence:.2%})")
