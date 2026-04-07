import os
import sys
import platform
import subprocess
import huggingface_hub
import json
import threading
import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import textwrap
import pyttsx3 

# Monkey patch
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline
from datasets import load_dataset

print("Initializing CogniBridge AI... (Loading dataset and model)")

# 1. Load the dataset
dataset = load_dataset("waboucay/wikilarge", 'original', split="train")

ex1_complex = dataset[0]['complex']
ex1_simple = dataset[0]['simple']
ex2_complex = dataset[1]['complex']
ex2_simple = dataset[1]['simple']
ex3_complex = dataset[2]['complex']
ex3_simple = dataset[2]['simple']

# 2. Custom PARAGRAPH
ex4_paragraph_complex = "In the event that the Purchaser fails to remit payment in full within the stipulated timeframe of thirty (30) days from the date of invoice issuance, the Vendor reserves the explicit right to suspend all ongoing services and impose a late penalty fee of one and one-half percent (1.5%) per month on the outstanding balance. Furthermore, any subsequent legal costs incurred during the collection process shall be borne entirely by the Purchaser."
ex4_paragraph_simple = "If the buyer doesn't pay the full money in 30 days after getting the invoice, the seller can pause all services and charge a 1.5% monthly fee on the unpaid money. The buyer must pay for any legal fees needed to collect the money."

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B-Instruct",
    dtype=ms.float32
)

print("CogniBridge is ready!\n")

# --- DYNAMIC CROSS-PLATFORM PATH FINDERS ---

def get_mindocr_python_path():
    """Dynamically finds the mindocr_env python executable based on the current cogni39 path."""
    current_python = sys.executable 
    # Swap the environment name in the path. This works identically on Mac, Windows, and Linux.
    return current_python.replace("cogni39", "mindocr_env")

def get_system_font_path():
    """Returns a valid bold font path depending on the current Operating System."""
    os_name = platform.system()
    if os_name == "Windows":
        return "C:/Windows/Fonts/arialbd.ttf"
    elif os_name == "Darwin": # macOS
        return "/System/Library/Fonts/Helvetica.ttc"
    else: # Linux / Raspberry Pi
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# -------------------------------------------

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
    return raw_output.split("Complex:")[0].strip()

def run_mindocr_isolated(image_path):
    mindocr_python_path = get_mindocr_python_path()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    predict_script = os.path.join(project_root, "mindocr", "tools", "infer", "text", "predict_system.py")
    
    command = [
        mindocr_python_path, predict_script,
        "--image_dir", image_path,
        "--det_algorithm", "DB++",
        "--rec_algorithm", "CRNN"
    ]
    
    subprocess.run(command, cwd=project_root)
    results_file = os.path.join(project_root, "inference_results", "system_results.txt")
    extracted_text = ""
    global_bbox = None
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) > 1:
                    raw_json_string = parts[1].strip()
                    try:
                        data = json.loads(raw_json_string)
                        processed_boxes = []
                        
                        g_min_x, g_min_y = float('inf'), float('inf')
                        g_max_x, g_max_y = 0, 0

                        for item in data:
                            points = item['points']
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_x = max(p[0] for p in points)
                            max_y = max(p[1] for p in points)
                            
                            g_min_x = min(g_min_x, min_x)
                            g_min_y = min(g_min_y, min_y)
                            g_max_x = max(g_max_x, max_x)
                            g_max_y = max(g_max_y, max_y)

                            processed_boxes.append({
                                'text': item['transcription'],
                                'min_x': min_x, 'center_y': (min_y + max_y) / 2.0, 'height': max_y - min_y
                            })
                        
                        if data:
                            global_bbox = (int(g_min_x)-10, int(g_min_y)-10, int(g_max_x)+10, int(g_max_y)+10)

                        processed_boxes.sort(key=lambda b: b['center_y'])
                        lines, current_line = [], []
                        for box in processed_boxes:
                            if not current_line:
                                current_line.append(box)
                            else:
                                prev_box = current_line[-1]
                                if abs(box['center_y'] - prev_box['center_y']) < max(box['height'], prev_box['height']) * 0.5:
                                    current_line.append(box)
                                else:
                                    lines.append(current_line)
                                    current_line = [box]
                        if current_line:
                            lines.append(current_line)
                            
                        ordered_words = []
                        for line_group in lines:
                            line_group.sort(key=lambda b: b['min_x'])
                            for box in line_group:
                                ordered_words.append(box['text'])
                        extracted_text += " ".join(ordered_words) + " "
                    except json.JSONDecodeError:
                        extracted_text += raw_json_string + " "
                        
    return extracted_text.strip(), global_bbox


# ==========================================
# THE GUI APPLICATION (AR CAMERA KIOSK)
# ==========================================
class CogniBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CogniBridge AI Scanner")
        
        # Keep fullscreen for Pi, but allow Mac/Windows testing to be easier to exit
        if platform.system() == "Linux":
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("800x600") # Windowed mode for desktop testing
            
        self.root.configure(bg="#000000") 
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.photos_dir = os.path.join(self.script_dir, "..", "data", "photos")
        os.makedirs(self.photos_dir, exist_ok=True) 
        
        self.frozen = False
        self.current_frame = None

        self.btn_font = font.Font(family="Helvetica", size=20, weight="bold")

        # Initialize Camera
        # macOS sometimes prefers cv2.CAP_AVFOUNDATION, but 0 is usually safe
        self.cap = cv2.VideoCapture(0)
        
        # Initialize the TTS Engine safely
        self.init_tts_engine()

        self.setup_ui()
        self.update_video_feed()

    def init_tts_engine(self):
        """Sets up the TTS engine and prepares the callback for your future UI highlighting."""
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160) 
        
        # PREP FOR FUTURE FEATURE: The 'onWord' callback
        # This fires every time the engine speaks a new word!
        def on_word(name, location, length):
            # print(f"Currently speaking word at index: {location}")
            # Later, you will update the Tkinter canvas here
            pass
            
        self.tts_engine.connect('onWord', on_word)

    def setup_ui(self):
        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.btn_frame = tk.Frame(self.root, bg="#000000")
        self.btn_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

        self.btn_action = tk.Button(self.btn_frame, text="📸 SCAN DOCUMENT", font=self.btn_font, 
                                  bg="#a6e3a1", fg="black", height=2, width=20, 
                                  command=self.handle_button_click)
        self.btn_action.pack(side=tk.LEFT, padx=20)

        self.btn_exit = tk.Button(self.btn_frame, text="❌ EXIT", font=self.btn_font, 
                                  bg="#f38ba8", fg="black", height=2, width=10, 
                                  command=self.on_exit)
        self.btn_exit.pack(side=tk.LEFT, padx=20)

    def update_video_feed(self):
        if not self.frozen:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                
                # Resize image slightly if not fullscreen to fit the testing window
                if platform.system() != "Linux":
                     pil_img = pil_img.resize((800, 600), Image.Resampling.LANCZOS)
                     
                self.photo = ImageTk.PhotoImage(image=pil_img)
                self.video_label.config(image=self.photo)

        self.root.after(15, self.update_video_feed)

    def handle_button_click(self):
        if not self.frozen:
            # Scanning mode
            self.frozen = True 
            filepath = os.path.join(self.photos_dir, "scan.png")
            cv2.imwrite(filepath, self.current_frame)
            
            self.btn_action.config(text="👁️ READING TEXT...", bg="#f9e2af", state=tk.DISABLED)
            threading.Thread(target=self.run_full_pipeline, args=(filepath,), daemon=True).start()
        else:
            # Resetting mode
            self.frozen = False
            self.btn_action.config(text="📸 SCAN DOCUMENT", bg="#a6e3a1", state=tk.NORMAL)
            
            # Stop the TTS engine if they reset early
            self.tts_engine.stop()

    def run_full_pipeline(self, filepath):
        raw_text, bbox = run_mindocr_isolated(filepath)
        
        if not raw_text or not bbox:
            self.root.after(0, self.update_button_state, "❌ NO TEXT FOUND - TAP TO RESET", "#f38ba8", tk.NORMAL)
            return

        self.root.after(0, self.update_button_state, "🧠 SIMPLIFYING...", "#cba6f7", tk.DISABLED)
        simplified = cognibridge_simplify(raw_text)
        
        self.root.after(0, self.draw_ar_overlay, filepath, simplified, bbox)
        
        # Fire the speech in a thread
        threading.Thread(target=self.speak_text, args=(simplified,), daemon=True).start()

    def speak_text(self, text):
        """Reads the text aloud using the OS-native offline voice."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def update_button_state(self, text, color, state):
        self.btn_action.config(text=text, bg=color, state=state)

    def draw_ar_overlay(self, filepath, simplified_text, bbox):
        img = Image.open(filepath).convert("RGBA")
        
        # Scale bounding box if image was resized for desktop testing
        if platform.system() != "Linux":
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1

        draw.rectangle(((x1, y1), (x2, y2)), fill=(30, 30, 46, 220))

        try:
            # Dynamically pull the correct OS font!
            font_path = get_system_font_path()
            ar_font = ImageFont.truetype(font_path, 20)
        except IOError:
            ar_font = ImageFont.load_default()

        chars_per_line = max(15, box_width // 11) 
        wrapped_text = textwrap.fill(simplified_text, width=chars_per_line)

        draw.multiline_text((x1 + 10, y1 + 10), wrapped_text, fill=(166, 227, 161, 255), font=ar_font, spacing=4)

        final_img = Image.alpha_composite(img, overlay).convert("RGB")
        self.photo = ImageTk.PhotoImage(image=final_img)
        self.video_label.config(image=self.photo)

        self.btn_action.config(text="🔄 TAP TO RESET", bg="#89b4fa", state=tk.NORMAL)

    def on_exit(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CogniBridgeApp(root)
    root.mainloop()