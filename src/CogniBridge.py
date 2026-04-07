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
<<<<<<< Updated upstream
import time
=======
>>>>>>> Stashed changes
import re

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
    current_python = sys.executable 
    return current_python.replace("cogni39", "mindocr_env")

def get_system_font_path():
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

Now, shorten and simplify this text using the simple sentence style. ONLY output the simplified text. Do not add any explanations, notes, or extra text. Do not write anything other than the simplified version of the text. Keep all the original meaning, but make it as easy to read as possible. Use everyday language and insert punctuation marks (. , ? !) where necessary.

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

def split_into_sentences(text):
    """
    Splits a block of text into individual sentences using standard punctuation.
    Perfect for step-by-step TTS playback.
    """
    if not text:
        return []
        
    text = text.strip()
    # Splits at spaces that immediately follow a period, exclamation, or question mark
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up the output to ensure no empty strings sneak through
    return [s.strip() for s in sentences if s.strip()]

# ==========================================
# THE GUI APPLICATION (AR CAMERA KIOSK)
# ==========================================
class CogniBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CogniBridge AI Scanner")

        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#000000") 
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.photos_dir = os.path.join(self.script_dir, "..", "data", "photos")
        os.makedirs(self.photos_dir, exist_ok=True) 
        
        self.frozen = False
        self.current_frame = None

        # --- NEW: Playback State Variables ---
        self.sentences = []
        self.current_sentence_index = 0
<<<<<<< Updated upstream
        self.is_paused = False
        self.stop_playback = False
=======
        self.is_playing = False
>>>>>>> Stashed changes

        self.btn_font = font.Font(family="Helvetica", size=20, weight="bold")
        self.text_font = font.Font(family="Helvetica", size=16)

        self.cap = cv2.VideoCapture(0)
        
        self.init_tts_engine()
        self.setup_ui()
        self.update_video_feed()
        
    def on_rewind(self):
        if self.current_sentence_index > 0:
            self.current_sentence_index -= 1

<<<<<<< Updated upstream
    def on_play_pause(self):
        # If the user presses play, but we are already at the end of the text, restart from the beginning
        if self.current_sentence_index >= len(self.sentences):
            self.current_sentence_index = 0
            
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.btn_play_pause.config(text="▶️") # Show Play
        else:
            self.btn_play_pause.config(text="⏸️") # Show Pause

    def on_fast_forward(self):
        # Allow fast-forwarding to the very end, which will trigger the auto-pause state
        if self.current_sentence_index < len(self.sentences):
            self.current_sentence_index += 1
=======
    def on_media_reverse(self):
        if self.sentences and self.current_sentence_index > 0:
            self.current_sentence_index -= 1
            print(f"\n◀️ REWIND TO SENTENCE {self.current_sentence_index}:")
            print(self.sentences[self.current_sentence_index])

    def on_media_play_pause(self):
        if not self.sentences:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.config(text="⏸️")
            print(f"\n▶️ RESUMING AT SENTENCE {self.current_sentence_index}:")
            print(self.sentences[self.current_sentence_index])
        else:
            self.btn_play_pause.config(text="▶️")
            print("\n⏸️ PAUSED PLAYBACK")

    def on_media_fast_forward(self):
        if self.sentences and self.current_sentence_index < len(self.sentences) - 1:
            self.current_sentence_index += 1
            print(f"\n⏩ SKIPPED TO SENTENCE {self.current_sentence_index}:")
            print(self.sentences[self.current_sentence_index])
>>>>>>> Stashed changes

    def init_tts_engine(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160) 
        
        self.voices = self.tts_engine.getProperty('voices')
        for voice in self.voices:
            if 'en' in voice.id.lower() or 'english' in voice.name.lower() or 'sam' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break

    def setup_ui(self):
        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.btn_settings = tk.Button(self.root, text="⚙️", font=font.Font(size=24), 
                                      bg="#1e1e2e", fg="white", bd=0, 
                                      command=self.open_settings)
        self.btn_settings.place(x=20, y=20) 

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

        self.playback_frame = tk.Frame(self.root, bg="#000000")
        
        self.btn_rewind = tk.Button(self.playback_frame, text="◀️", font=self.btn_font, 
                                    bg="#89b4fa", fg="black", height=1, width=4, 
                                    command=self.on_rewind)
        self.btn_rewind.pack(side=tk.LEFT, padx=10)

        self.btn_play_pause = tk.Button(self.playback_frame, text="⏸️", font=self.btn_font, 
                                        bg="#f9e2af", fg="black", height=1, width=4, 
                                        command=self.on_play_pause)
        self.btn_play_pause.pack(side=tk.LEFT, padx=10)

        self.btn_fast_forward = tk.Button(self.playback_frame, text="⏩", font=self.btn_font, 
                                          bg="#89b4fa", fg="black", height=1, width=4, 
                                          command=self.on_fast_forward)
        self.btn_fast_forward.pack(side=tk.LEFT, padx=10)

    def open_settings(self):
        self.frozen = True 
        self.settings_frame = tk.Frame(self.root, bg="#1e1e2e", bd=5, relief=tk.RAISED)
        self.settings_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.6, relheight=0.4)
        
        tk.Label(self.settings_frame, text="⚙️ Text-to-Speech Settings", font=self.btn_font, bg="#1e1e2e", fg="#cdd6f4").pack(pady=20)
        
        voice_names = [v.name for v in self.voices]
        self.selected_voice_name = tk.StringVar(self.root)
        
        current_id = self.tts_engine.getProperty('voice')
        current_name = next((v.name for v in self.voices if v.id == current_id), voice_names[0])
        self.selected_voice_name.set(current_name)
        
        dropdown = tk.OptionMenu(self.settings_frame, self.selected_voice_name, *voice_names)
        dropdown.config(font=self.text_font, bg="#313244", fg="#cdd6f4", highlightthickness=0)
        dropdown.pack(pady=20)
        
        def save_and_close():
            chosen = self.selected_voice_name.get()
            for v in self.voices:
                if v.name == chosen:
                    self.tts_engine.setProperty('voice', v.id)
                    break
            self.settings_frame.destroy()
            self.frozen = False 
            
        tk.Button(self.settings_frame, text="✅ Save & Close", font=self.btn_font, bg="#a6e3a1", fg="black", command=save_and_close).pack(pady=20)

    def update_video_feed(self):
        if not self.frozen:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                self.photo = ImageTk.PhotoImage(image=pil_img)
                self.video_label.config(image=self.photo)

        self.root.after(15, self.update_video_feed)

    def handle_button_click(self):
        if not self.frozen:
            self.frozen = True 
            filepath = os.path.join(self.photos_dir, "scan.png")
            cv2.imwrite(filepath, self.current_frame)
            
            self.btn_action.config(text="👁️ READING TEXT...", bg="#f9e2af", state=tk.DISABLED)
            threading.Thread(target=self.run_full_pipeline, args=(filepath,), daemon=True).start()
        else:
            # --- NEW: Stop playback logic when resetting ---
            self.stop_playback = True
            
            self.frozen = False
            self.btn_action.config(text="📸 SCAN DOCUMENT", bg="#a6e3a1", state=tk.NORMAL)
            self.playback_frame.place_forget()

    def run_full_pipeline(self, filepath):
        raw_text, bbox = run_mindocr_isolated(filepath)
        
        if not raw_text or not bbox:
            self.root.after(0, self.update_button_state, "❌ NO TEXT FOUND - TAP TO RESET", "#f38ba8", tk.NORMAL)
            return

        self.root.after(0, self.update_button_state, "🧠 SIMPLIFYING...", "#cba6f7", tk.DISABLED)
        simplified = cognibridge_simplify(raw_text)
        
        #self.root.after(0, self.draw_ar_overlay, filepath, simplified, bbox)
        #threading.Thread(target=self.speak_text, args=(simplified,), daemon=True).start()

        # --- NEW: Split the text and reset trackers ---
        self.sentences = split_into_sentences(simplified)
        self.current_sentence_index = 0
        self.is_playing = True
        
        self.root.after(0, self.draw_ar_overlay, filepath, simplified, bbox)
        
        # --- NEW: Call the new start_playback method ---
        self.start_playback(simplified)

    def start_playback(self, text):
        # Clean text and split by punctuation (. ! ?)
        raw_sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
        self.sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        self.current_sentence_index = 0
        self.is_paused = False
        self.stop_playback = False
        self.btn_play_pause.config(text="⏸️") # Reset button to pause icon

        # Start background reading loop
        threading.Thread(target=self.speak_loop, daemon=True).start()

    def speak_loop(self):
        """Background thread that stays alive until reset, handling TTS reading."""
        while not self.stop_playback:
            # If paused, OR if we've reached the end of the text, just wait/idle here.
            if self.is_paused or self.current_sentence_index >= len(self.sentences):
                
                # Auto-pause the UI when reaching the end of the text naturally
                if self.current_sentence_index >= len(self.sentences) and not self.is_paused:
                    self.is_paused = True
                    self.root.after(0, lambda: self.btn_play_pause.config(text="▶️"))
                
                time.sleep(0.1)
                continue

            # Record index before speaking
            idx_before_speaking = self.current_sentence_index
            
            # Speak the current sentence (Blocks until finished)
            self.tts_engine.say(self.sentences[self.current_sentence_index])
            self.tts_engine.runAndWait()

            # Move to the next sentence ONLY if the user didn't manually skip/rewind while it was talking
            if not self.stop_playback and not self.is_paused:
                if self.current_sentence_index == idx_before_speaking:
                    self.current_sentence_index += 1
        # Reset UI if playback finishes naturally
        if not self.stop_playback:
            self.root.after(0, lambda: self.btn_play_pause.config(text="▶️"))

    def update_button_state(self, text, color, state):
        self.btn_action.config(text=text, bg=color, state=state)

    def draw_ar_overlay(self, filepath, simplified_text, bbox):
        img = Image.open(filepath).convert("RGBA")
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1

        draw.rectangle(((x1, y1), (x2, y2)), fill=(30, 30, 46, 220))

        try:
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
        
        self.playback_frame.place(relx=0.5, rely=0.78, anchor=tk.CENTER)

    def on_exit(self):
        self.stop_playback = True
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CogniBridgeApp(root)
    root.mainloop()