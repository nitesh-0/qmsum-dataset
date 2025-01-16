import os 
import json

# Specify the directory containing the JSON files
input_folder = r"D:\dataset\QSUM\QMSum\data\ALL\val"
output_folder = r"D:\dataset\QSUM\QMSum\data\ALL\valProcessed"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        
        # Load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # Extract and concatenate the meeting transcript
        transcript_content = " ".join(
            entry["content"] for entry in data["meeting_transcripts"]
        )
        
        # Extract the summary (answer of the first query in the array)
        summary = data["general_query_list"][0]["answer"] if data["general_query_list"] else None
        
        # Create the output format
        output = [{"transcript": transcript_content, "summarize": summary}]
        
        # Save the transformed data to the output folder
        output_file_path = os.path.join(output_folder, f"processed_{filename}")
        with open(output_file_path, "w") as output_file:
            json.dump(output, output_file, indent=4)
        
        print(f"Processed: {filename} -> {output_file_path}")

print("All files have been processed.")
