import re
import huggingface_hub

# Our trusty monkey patch to keep the pipeline happy
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline

print("Loading BERT... (This might take a moment on the CPU)")
# Initialize the fill-mask pipeline
pipe = pipeline(
    "fill-mask", 
    model="bert-base-uncased", 
    torch_dtype=ms.float32  # Keeping your Mac CPU safe!
)

def simplify_sentence_with_bert(sentence, threshold_length=7):
    # 1. Split the sentence into words and punctuation
    tokens = re.findall(r'\b\w+\b|[.,!?;]', sentence)
    simplified_tokens = []

    for i, token in enumerate(tokens):
        # 2. If the word is long enough, mask it and ask BERT for help
        if token.isalpha() and len(token) >= threshold_length:
            
            # Create a temporary copy of the sentence with the [MASK] token
            temp_tokens = tokens.copy()
            temp_tokens[i] = "[MASK]"
            
            # Rebuild the sentence as a string to feed into BERT
            masked_sentence = " ".join(temp_tokens).replace(" ,", ",").replace(" .", ".")
            
            # Ask BERT for the top 5 predictions
            predictions = pipe(masked_sentence, top_k=5)
            
            # 3. Pick the best prediction that ISN'T the original word
            best_replacement = token 
            for pred in predictions:
                candidate = pred['token_str'].strip()
                
                # Make sure BERT doesn't just suggest the exact same word again
                if candidate.lower() != token.lower() and candidate.isalpha():
                    best_replacement = candidate
                    break  # Stop at the first good, different word
            
            simplified_tokens.append(best_replacement)
            print(f"Replaced '{token}' -> '{best_replacement}'")
        else:
            # If it's a short word or punctuation, leave it alone
            simplified_tokens.append(token)

    # 4. Stitch the final sentence back together
    final_sentence = " ".join(simplified_tokens).replace(" ,", ",").replace(" .", ".")
    return final_sentence

# Let's test it out!
original_text = "The physician will ameliorate the excruciating discomfort."

print("\n--- Starting Simplification ---")
print(f"Original: {original_text}")

simplified_text = simplify_sentence_with_bert(original_text)

print(f"\nFinal Result: {simplified_text}")