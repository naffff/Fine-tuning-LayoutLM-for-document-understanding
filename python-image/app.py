from PIL import Image, ImageDraw, ImageFont
from functools import partial
import numpy as np
import torch
from datasets import Features, Sequence, ClassLabel, Value, Array2D, load_dataset
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification, Trainer, TrainingArguments
import evaluate
from huggingface_hub import HfFolder

# Define the model and dataset IDs
processor_id = "microsoft/layoutlmv2-base-uncased"
dataset_id = "nielsr/funsd"

# Load the dataset
dataset = load_dataset(dataset_id)

# Define the labels for NER tagging
labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']

# Create the LayoutLMv2 processor
processor = LayoutLMv2Processor.from_pretrained(processor_id, apply_ocr=False)

# Define custom features for the dataset
features = Features(
    {
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(Value(dtype="int64")),
        "token_type_ids": Sequence(Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

# Preprocess function to prepare the data in the correct format for the model
def process(sample, processor=None):
    encoding = processor(
        Image.open(sample["image_path"]).convert("RGB"),
        sample["words"],
        boxes=sample["bboxes"],
        word_labels=sample["ner_tags"],
        padding="max_length",
        truncation=True,
    )
    del encoding["image"]
    return encoding

# Process the dataset and format it for PyTorch
proc_dataset = dataset.map(
    partial(process, processor=processor),
    remove_columns=["image_path", "words", "ner_tags", "id", "bboxes"],
    features=features,
).with_format("torch")

# Load the LayoutLMForTokenClassification model
model_id = "microsoft/layoutlm-base-uncased"
model = LayoutLMForTokenClassification.from_pretrained(
    model_id, num_labels=len(labels), label2id={k: v for v, k in enumerate(labels)}, id2label={v: k for v, k in enumerate(labels)}
)

# Load seqeval metric for evaluation
metric = evaluate.load("seqeval")
ner_labels = list(model.config.id2label.values())

# Function to compute metrics during evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    all_predictions = []
    all_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])
    return metric.compute(predictions=[all_predictions], references=[all_labels])

# Define training arguments
repository_id = "layoutlm-funsd"
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    fp16=False,
    learning_rate=3e-5,
    logging_dir=f"{repository_id}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc_dataset["train"],
    eval_dataset=proc_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Change apply_ocr to True to use the OCR text for inference
processor.feature_extractor.apply_ocr = True

# Save the processor and create a model card
processor.save_pretrained(repository_id)
trainer.create_model_card()
trainer.push_to_hub()

# Function to unnormalize bounding boxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

# Label-to-color mapping for drawing bounding boxes
label2color = {
    "B-HEADER": "blue",
    "B-QUESTION": "red",
    "B-ANSWER": "green",
    "I-HEADER": "blue",
    "I-QUESTION": "red",
    "I-ANSWER": "green",
}

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # Draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

# Function to run inference and optionally draw bounding boxes on the image
def run_inference(path, model=model, processor=processor, output_image=True):
    # Create model input
    image = Image.open(path).convert("RGB")
    encoding = processor(image, return_tensors="pt")
    del encoding["image"]
    # Run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # Get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels)
    else:
        return labels

# Example: Run inference on a specific image in the test dataset
inference_image_path = dataset["test"][34]["image_path"]
resulting_image = run_inference(inference_image_path)
resulting_image.show()
