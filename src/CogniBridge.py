import os
import subprocess
import huggingface_hub
import json                 # for parsing the MindOCR JSON output and extracting the transcription values before handing the text over to Qwen

# Monkey patch, because the default Hugging Face cache directory is not writable in this environment. This will force it to download files directly without caching, which is fine for our use case
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline
from datasets import load_dataset

print("Initializing CogniBridge AI... (Loading dataset and model)")

# 1. Load the dataset for our short sentence examples
dataset = load_dataset("waboucay/wikilarge", 'original', split="train")

ex1_complex = dataset[0]['complex']
ex1_simple = dataset[0]['simple']
ex2_complex = dataset[1]['complex']
ex2_simple = dataset[1]['simple']
ex3_complex = dataset[2]['complex']
ex3_simple = dataset[2]['simple']
ex4_complex = dataset[3]['complex']
ex4_simple = dataset[3]['simple']

# 2. Our custom PARAGRAPH example to teach it how to handle long legal text
ex4_paragraph_complex = "In the event that the Purchaser fails to remit payment in full within the stipulated timeframe of thirty (30) days from the date of invoice issuance, the Vendor reserves the explicit right to suspend all ongoing services and impose a late penalty fee of one and one-half percent (1.5%) per month on the outstanding balance. Furthermore, any subsequent legal costs incurred during the collection process shall be borne entirely by the Purchaser."

ex4_paragraph_simple = "If the buyer doesn't pay the full money in 30 days after getting the invoice, the seller can pause all services and charge a 1.5% monthly fee on the unpaid money. The buyer must pay for any legal fees needed to collect the money."

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B-Instruct",
    dtype=ms.float32
)

print("CogniBridge is ready!\n")

def cognibridge_simplify(text):
    word_count = len(text.split())
    dynamic_max_tokens = min(int((word_count * 2) + 20), 512)
    prompt = f"""You are an expert at simplifying complex English. Look at these examples of complex text being rewritten into simple text.

Complex: {ex1_complex}
Simple: {ex1_simple}

Complex: {ex2_complex}
Simple: {ex2_simple}

Complex: {ex3_complex}
Simple: {ex3_simple}

Complex: {ex4_paragraph_complex}
Simple: {ex4_paragraph_simple}

Now, shorten and simplify this text using the simple sentence style. ONLY output the simplified text. Do not add any explanations, notes, or extra text.
Complex: {text}
Simple:"""

    result = pipe(
        prompt,
        max_new_tokens=dynamic_max_tokens,
        return_full_text=False,
        temperature=0.1
    )

    raw_output = result[0]["generated_text"].strip()
    clean_output = raw_output.split("Complex:")[0].strip()
    return clean_output

def process_document(input_filename, output_filename):
    """Reads a text file line-by-line, simplifies it, and saves the output."""
    if not os.path.exists(input_filename):
        print(f"Error: I couldn't find '{input_filename}'.")
        return

    print(f"Opening '{input_filename}' for processing...\n")
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, 1):
            original_sentence = line.strip()
            if not original_sentence:
                continue
            print(f"Simplifying sentence {line_number}...")
            simplified = cognibridge_simplify(original_sentence)
            outfile.write(f"Original: {original_sentence}\n")
            outfile.write("-" * 50 + "\n\n")
            outfile.write(f"Simplified: {simplified}\n")
            
    print(f"\nSuccess! All simplified sentences have been saved to '{output_filename}'")

def run_mindocr_isolated(image_path):
    """Reaches into the isolated mindocr_env to read text from an image."""
    print(f"Running MindOCR on '{image_path}'...")
    mindocr_python_path = "/opt/miniconda3/envs/mindocr_env/bin/python"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    predict_script = os.path.join(project_root, "mindocr", "tools", "infer", "text", "predict_system.py")
    
    command = [
        mindocr_python_path,
        predict_script,
        "--image_dir", image_path,
        "--det_algorithm", "DB++",
        "--rec_algorithm", "CRNN"
    ]
    
    subprocess.run(command, cwd=project_root)
    results_file = os.path.join(project_root, "inference_results", "system_results.txt")
    extracted_text = ""
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) > 1:
                    raw_json_string = parts[1].strip()
                    try:
                        # Parse the JSON array
                        data = json.loads(raw_json_string)
                        
                        # 1. Extract coordinates and calculate center Y, min X, and height
                        processed_boxes = []
                        for item in data:
                            points = item['points']
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_y = max(p[1] for p in points)
                            
                            processed_boxes.append({
                                'text': item['transcription'],
                                'min_x': min_x,
                                'center_y': (min_y + max_y) / 2.0,
                                'height': max_y - min_y
                            })
                            
                        # 2. Sort all boxes primarily top-to-bottom
                        processed_boxes.sort(key=lambda b: b['center_y'])
                        
                        # 3. Group the boxes into horizontal lines
                        lines = []
                        current_line = []
                        
                        for box in processed_boxes:
                            if not current_line:
                                current_line.append(box)
                            else:
                                prev_box = current_line[-1]
                                # If the vertical distance is less than half the text height, 
                                # we consider them to be on the same line.
                                if abs(box['center_y'] - prev_box['center_y']) < max(box['height'], prev_box['height']) * 0.5:
                                    current_line.append(box)
                                else:
                                    lines.append(current_line)
                                    current_line = [box]
                                    
                        if current_line:
                            lines.append(current_line)
                            
                        # 4. Sort each line from left-to-right and combine
                        ordered_words = []
                        for line_group in lines:
                            line_group.sort(key=lambda b: b['min_x'])
                            for box in line_group:
                                ordered_words.append(box['text'])
                                
                        # Join the final words into a single string
                        extracted_text += " ".join(ordered_words) + " "
                        
                    except json.JSONDecodeError:
                        # Fallback just in case the output isn't valid JSON
                        extracted_text += raw_json_string + " "
                    
    return extracted_text.strip()

def process_image(image_path, output_filename):
    """Extracts text from an image using MindOCR, simplifies it, and saves it."""
    if not os.path.exists(image_path):
        print(f"Error: I couldn't find '{image_path}'.")
        return

    raw_text = run_mindocr_isolated(image_path)
    if not raw_text:
        print("No text was found in the image, or OCR failed.")
        return
        
    print(f"\n--- Extracted Text from Image ---\n{raw_text}\n")
    print("Simplifying with Qwen...")
    simplified = cognibridge_simplify(raw_text)
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Original (From Image): {raw_text}\n")
        outfile.write("-" * 50 + "\n\n")
        outfile.write(f"Simplified: {simplified}\n")
        
    print(f"\nSuccess! The image text has been simplified and saved to '{output_filename}'")


# --- How to run the pipeline ---
if __name__ == "__main__":
    # Dynamically find where this script is living
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup paths for TEXT processing
    input_txt = os.path.join(script_dir, "..", "data", "complex_text.txt")
    output_txt = os.path.join(script_dir, "..", "data", "simplified_text.txt")
    
    # Setup paths for IMAGE processing
    input_img = os.path.join(script_dir, "..", "data", "scan.png")
    output_img_results = os.path.join(script_dir, "..", "data", "simplified_image.txt")
    
    # ---------------------------------------------------------
    # CHOOSE WHAT TO RUN HERE:
    # Just comment (#) or uncomment the one you want to test!
    # ---------------------------------------------------------
    
    # Test 1: Process your text document
    process_document(input_txt, output_txt)
    
    # Test 2: Process your scanned image
    #process_image(input_img, output_img_results)