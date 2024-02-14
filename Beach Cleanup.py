# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
os.environ['OPENAI_API_KEY'] = ''

import os
import json
from openai import OpenAI
import random
from datetime import datetime, timedelta
import re

# ### Generate synthetic data using GPT

# +
# Sample lists of data points
organizations = [
    "Ocean Cleanup Initiative", "Marine Life Guardians", "Eco Ocean Warriors", "Sea Savers",
    "Beach Protectors", "Wave Watchers", "Tide Turners", "Coastal Caretakers"
]
locations = [
    "Coral Bay", "Blue Reef", "Marina Beach", "Sunset Shore", "Pelican Point", "Dolphin Cove",
    "Seagull Island", "Mystic Beach"
]
trash_types = [
    "Plastic Waste", "Metal Debris", "Assorted Trash", "Assorted Trash", "Industrial Waste", "Chemical Containers", 
    "Fishing Nets", "Glass Bottles", "Electronic Waste", "Rubber Tyres", "Textile Scraps"
]
types = ["instagram", "press_release"]
start_date = datetime(2017, 1, 1)

# Function to generate a random date
def generate_random_date():
    random_days = random.randint(0, 365 * 5)  # Random date within 5 years
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

# Function to generate a random data point
def generate_data_point():
    return {
        "weight": str(random.randint(100, 500)),  # Random weight between 100 and 500
        "organization": random.choice(organizations),
        "date": generate_random_date(),
        "location": random.choice(locations),
        "trash_type": random.choice(trash_types),
        "type": random.choice(types)
    }

# Generate multiple data points
num_data_points = 1000  # Adjust the number of data points as needed
data_points = [generate_data_point() for _ in range(num_data_points)]

# Save to a JSON file
with open('data/data_points.json', 'w') as file:
    json.dump(data_points, file, indent=4)

# +
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Define the structure of your prompts
prompt_structure = {
    "instagram": "Generate an Instagram caption for a beach cleanup where {weight} kilograms of {trash_type} were cleaned.\nOrganization: {organization}\nDate: {date}\nLocation: {location}",
    "press_release": "Generate a press release for a beach cleanup where {weight} kilograms of {trash_type} were cleaned.\nOrganization: {organization}\nDate: {date}\nLocation: {location}"
}

# Define the data points you want to use to generate the reports
file_path = 'data/data_points.json'

# Load the data from the JSON file
with open(file_path, 'r') as file:
    data_points = json.load(file)

def remove_unicode_characters(text):
    # Regex to match non-ASCII characters
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    # Remove non-ASCII characters
    text = non_ascii_pattern.sub('', text)
    # Replace newline characters with a space 
    text = text.replace('\n', ' ')
    text = text.replace('\"', ' ')
    return text



# +
# Function to generate data
def generate_data(data_points, prompt_structure, num_records):   
    generated_data = []
    for _ in range(num_records):
            point = random.choice(data_points)
            prompt_type = point['type']
            prompt = prompt_structure[prompt_type].format(**point)

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=64
            )
            generated_text = remove_unicode_characters(response.choices[0].message.content.strip())
            # Format the response into your desired structure
            record = {
                "date": point["date"],
                "location": point["location"],
                "organization": point["organization"],
                "weight_kg": point["weight"],
                "trash_type": point["trash_type"],  # Extract from generated_text or define
                "caption": generated_text
            }
            generated_data.append(record)

    return generated_data

# Generate and save the reports
synthetic_reports = generate_data(data_points, prompt_structure, 100)

filename = 'data/synthetic_reports.json'
with open(filename, 'w') as file:
    json.dump(synthetic_reports, file, indent=4)

print(f"Reports saved to {filename}")
# -

# ### Use a Pre-trained NER Model for Pre-annotation

import spacy

# +
# Load a pre-trained NER model
nlp = spacy.load("en_core_web_sm")

def pre_annotate(text):
    doc = nlp(text)
    labels = []
    for token in doc:
        if token.ent_iob_ != 'O':
            labels.append(f"{token.ent_iob_}-{token.ent_type_}")
        else:
            labels.append('O')
    return labels


# +
# Load your dataset
with open('data/synthetic_reports.json', 'r') as file:
    data = json.load(file)

# Apply pre-annotation
for item in data:
    item['labels'] = pre_annotate(item['caption'])

# Optionally, save the annotated data back to a file
with open('data/synthetic_reports_annotated.json', 'w') as file:
    json.dump(data, file, indent=4)
# -

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import string


# ### Dataset Class

class BeachCleanupDataset(Dataset):
    def __init__(self, json_file, vocab, label_to_idx):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.vocab = vocab
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        caption_tokens = tokenize(item['caption'])
        caption_indices = [self.vocab.get(token, 0) for token in caption_tokens]

        if len(item['labels']) != len(caption_tokens):
            print("Mismatch found in item:", idx)
            print("Caption:", item['caption'])
            print("Tokens:", caption_tokens)
            print("Labels:", item['labels'])
            raise AssertionError("Mismatch in tokens and labels length")

        labels = [self.label_to_idx[label] for label in item['labels']]

        return torch.tensor(caption_indices), torch.tensor(labels)


# ### Tokenization and Vocabulary Building

# +
# A simple tokenizer
def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Build a vocabulary from the dataset
def build_vocab(data):
    counter = Counter()
    for item in data:
        tokens = tokenize(item['caption'])
        counter.update(tokens)
    return {word: i+1 for i, (word, _) in enumerate(counter.most_common())}



# +
# Build the vocabulary
vocab = build_vocab(data)

# Define your labels based on the NER task
unique_labels = set()
for item in data:
    unique_labels.update(item['labels'])

# Now create label_to_idx with all unique labels
label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}


# -

# ### Model Definition

# +
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# -

from sklearn.metrics import classification_report
import numpy as np


# ### Define the Evaluation Function

def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for captions, labels in dataloader:
            outputs = model(captions)

            # Flatten outputs and labels
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions and true labels
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            valid_indices = labels != -1  # Indices where labels are not padding
            valid_predictions = predictions[valid_indices]
            valid_labels = labels[valid_indices]

            all_predictions.extend(valid_predictions)
            all_labels.extend(valid_labels)

    average_loss = total_loss / len(dataloader)

    # Ensure target_names match all labels in your dataset
    unique_labels = sorted(set(all_labels + all_predictions))  # Combine and sort to get all unique labels
    target_names = [label for label, idx in sorted(label_to_idx.items(), key=lambda item: item[1]) if idx in unique_labels]

    # Update the classification report call
    report = classification_report(all_labels, all_predictions, labels=unique_labels, target_names=target_names, zero_division=0)

    
    return average_loss, report


# ### Training Loop

def train_and_evaluate(model, train_dataloader, eval_dataloader, criterion, optimizer, num_epochs, log_interval=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0

        for batch_idx, (captions, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(captions)

            # Flatten outputs and labels
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            # Compute loss
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}, Training Loss: {loss.item():.4f}")

        average_train_loss = total_train_loss / len(train_dataloader)

        # Evaluate the model on the validation set
        eval_loss, eval_report = evaluate(model, eval_dataloader, criterion)

        # Log epoch-level training and evaluation information
        print(f"End of Epoch {epoch + 1}")
        print(f"Training Loss: {average_train_loss:.4f}")
        print(f"Evaluation Loss: {eval_loss:.4f}")
#         print("Evaluation Report:")
#         print(eval_report)



# #### Define a custom collate function for the DataLoader to handle variable-length sequences:

def collate_fn(batch):
    captions, labels = zip(*batch)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return captions_padded, labels_padded



from sklearn.model_selection import train_test_split

# +
# Load the entire dataset
with open('data/synthetic_reports_annotated.json', 'r') as file:
    full_dataset = json.load(file)

# Split the dataset into training and validation sets (80-20 split)
train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)

# Save the split datasets if needed
with open('data/train_data.json', 'w') as file:
    json.dump(train_data, file)

with open('data/val_data.json', 'w') as file:
    json.dump(val_data, file)

# +
# Initialize the training and validation datasets
train_dataset = BeachCleanupDataset('data/train_data.json', vocab, label_to_idx)
val_dataset = BeachCleanupDataset('data/val_data.json', vocab, label_to_idx)

# Initialize the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# +
# Define dimensions for your model
embedding_dim = 100  # Example dimension
hidden_dim = 64      # Example dimension
num_classes = len(label_to_idx)  # Number of unique NER labels

model = NERModel(len(vocab)+1, embedding_dim, hidden_dim, num_classes)

criterion = nn.CrossEntropyLoss(ignore_index=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
num_epochs = 20
train_and_evaluate(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)
# -

# Evauate the model
eval_loss, eval_report = evaluate(model, val_dataloader, criterion)
print(f"Evaluation Loss: {eval_loss}")
print("Evaluation Report:")
print(eval_report)



