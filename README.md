# 🌉 CogniBridge

CogniBridge is an AI-powered text simplification engine designed to translate complex English (including but not limited to legal, corporate, or academic documents) into plain, accessible language.

It leverages **MindSpore**, **MindNLP** (powering a local Qwen 2 Instruction model), and **MindOCR** to process both digital text and scanned physical documents.

## 🧠 Architecture

We use two separate Conda environments:
1. **`cogni39` (The NLP Brain):** Runs MindSpore 2.7.0 and handles the text-to-text generative AI using Qwen2-0.5B-Instruct.
2. **`mindocr_env` (The Vision Brain):** Runs MindSpore 2.5.0 and handles optical character recognition for scanned PDFs, JPEGs and PNGs.

The main script seamlessly bridges these environments by delegating OCR tasks to the isolated vision environment when necessary.

---

## ⚙️ Prerequisites
Before installing, ensure your system has the following:
* **Conda** (Miniconda suggested)\
Install on Debian through:\
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`\
For other systems: https://www.anaconda.com/docs/getting-started/miniconda/install
* **Git**
* **[MindSpore 2.5.0 wheel installation package](https://www.mindspore.cn/versions/en/)**

Tested on:
* macOS 26 (ARM64)
* Debian 13 (ARM64)

---

## 🚀 Installation Guide

**1. Clone the CogniBridge repository to your desired location**

`git clone https://github.com/bilusgarage/CogniBridge`

**2. Download [MindSpore 2.5.0 (Python 3.9) package](https://www.mindspore.cn/versions/en/#2.5.0) for your system**

Example:
* for Windows x86-64: `mindspore-2.5.0-cp39-cp39-win_amd64.whl`
* for Linux ARM: `mindspore-2.5.0-cp39-cp39-linux_aarch64.whl` 

**2. Move the `mindspore[...].whl` file to the path `CogniBridge/mindspore_installation_package/`**

**3. Launch installation script `install.py`**

## 🤔 How to use it?

* Simplifying a `.txt` document
	1. At the end of `src/CogniBridge.py` file uncomment
`process_document(input_txt, output_txt)`
	2. Comment
`process_image(input_img, output_img_results)`
	3. Put your text file as `data/complex_text.txt`
	4. `conda activate cogni39`
	5. Run `CogniBridge.py`
	6. Check results in `data/simplified_text.txt`

* Simplifying text from a `.png` documentq
	1. At the end of `src/CogniBridge.py` file uncomment
`process_image(input_img, output_img_results)`
	2. Comment
`process_document(input_txt, output_txt)`
	3. Put your PNG image file as `data/scan.png`
	4. `conda activate cogni39`
	5. Run `CogniBridge.py`
	6. Check results in `data/simplified_image.txt`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.