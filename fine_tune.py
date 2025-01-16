import json
import re
import os

# Function to clean a transcript
def clean_transcript(transcript):
    # Step 1: Remove filler words and markers
    patterns = [
        r"\b(uh|um|you know|like|okay|well|mm|true|yeah|oh|hmm|huh|hmmm|ah|so so|nee)\b",  # Common filler words
        r"\{.*?\}"  # Patterns like "{vocalsound}", "{gap}", "{disfmarker}"
    ]
    for pattern in patterns:
        transcript = re.sub(pattern, '', transcript, flags=re.IGNORECASE)

    # Step 2: Replace consecutive punctuation marks (e.g., "..", ",,", "!!!") with a single mark
    transcript = re.sub(r'[.,!?]+', lambda m: m.group(0)[0], transcript)

    # Step 3: Remove consecutive duplicate words (case-insensitive)
    transcript = re.sub(r'\b(\w+)\s+\1\b', r'\1', transcript, flags=re.IGNORECASE)

    # Step 4: Normalize spacing and ensure proper punctuation spacing
    transcript = re.sub(r'\s*([.,!?])\s*', r'\1 ', transcript)
    transcript = re.sub(r'\s+', ' ', transcript).strip()

    # Step 5: Remove leading or trailing punctuation
    transcript = re.sub(r'^[.,!?]+', '', transcript).strip()
    transcript = re.sub(r'[.,!?]+$', '', transcript).strip()

    return transcript

# Paths
input_folder = '/content/drive/MyDrive/Minor_project/processed'  # Update this path if necessary
output_folder = '/content/drive/MyDrive/Minor_project/cleaned'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.json'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Load and clean the JSON data
        with open(input_path, 'r') as file:
            data = json.load(file)

        # Replace transcript with the cleaned version
        for entry in data:
            if "transcript" in entry:
                entry["transcript"] = clean_transcript(entry["transcript"])

        # Save the cleaned data
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Processed and saved: {output_path}")

print("All files cleaned and saved!")


!pip install transformers datasets torch



import os
import json
from datasets import Dataset

# Path to the folder containing the JSON files
folder_path = '/content/drive/MyDrive/Minor_project/cleaned/'

# Load all JSON files in the folder
def load_json_files(folder_path):
    data = {'input': [], 'target': []}

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                # Load the content of the file
                content = json.load(f)

                # Extract 'transcript' and 'summarize' fields
                for entry in content:
                    data['input'].append(entry['transcript'])
                    data['target'].append(entry['summarize'])

    return data

# Load the dataset
data = load_json_files(folder_path)

# Convert the data into a Hugging Face Dataset
dataset = Dataset.from_dict(data)


from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Preprocessing function to tokenize the input and target text
def preprocess_function(examples):
    # Tokenize the inputs and labels, apply padding to max_length
    model_inputs = tokenizer(examples['input'], max_length=512, padding='max_length', truncation=True)
    labels = tokenizer(examples['target'], max_length=150, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# Apply the preprocessing function to the dataset
dataset = dataset.map(preprocess_function, batched=True)





from datasets import Dataset, DatasetDict

# Example loading dataset (replace 'data' with your actual data)
dataset = Dataset.from_dict(data)

# First, split into train and test sets (90% training, 10% testing)
train_val = dataset.train_test_split(test_size=0.1)
train_dataset = train_val['train']
test_dataset = train_val['test']

# Further split the training data into training and validation sets (90% training, 10% validation)
train_val_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Now, wrap the split datasets in a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})





# Chunking function to split tokenized sequences that exceed max_length
def chunk_sequence(sequence, max_length=512):
    # Split into smaller chunks if the sequence exceeds max_length
    return [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]

# Apply chunking to the dataset (if needed)
def chunk_function(examples):
    # Chunk the input sequences if they exceed max_length (512 for T5)
    for i in range(len(examples['input_ids'])):
        if len(examples['input_ids'][i]) > 512:
            examples['input_ids'][i] = chunk_sequence(examples['input_ids'][i])

        if len(examples['labels'][i]) > 150:  # Assuming the max length for summary is 150 tokens
            examples['labels'][i] = chunk_sequence(examples['labels'][i], max_length=150)

    return examples

# Apply chunking function on the dataset
dataset = dataset.map(chunk_function, batched=True)





from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',             # Output directory to save the model and logs
    eval_strategy="epoch",              # Evaluate after each epoch (use eval_strategy instead of evaluation_strategy)
    learning_rate=2e-5,                 # Learning rate for training
    per_device_train_batch_size=8,      # Batch size for training
    per_device_eval_batch_size=8,       # Batch size for evaluation
    num_train_epochs=3,                 # Number of training epochs
    weight_decay=0.01,                  # Weight decay for regularization
    logging_dir='./logs',               # Directory for storing logs
    logging_steps=500,                  # Log every 500 steps
    save_strategy="epoch",              # Save model after each epoch (matches eval_strategy)
    save_total_limit=2,                 # Limit the number of saved models to avoid clutter
    push_to_hub=False,                  # Set to True if you want to upload to Hugging Face hub
    load_best_model_at_end=True,        # Load the best model when finished training
)




from transformers import T5ForConditionalGeneration, Trainer

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Adjust model size as needed

# Prepare the dataset for Trainer (train and eval split)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Initialize Trainer with model, arguments, and datasets
trainer = Trainer(
    model=model,                             # The pre-trained model
    args=training_args,                      # Training arguments
    train_dataset=train_dataset,             # Training dataset
    eval_dataset=eval_dataset,               # Evaluation dataset
)



trainer.train()