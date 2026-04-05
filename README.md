# Prompt Injection Detector

An AI-powered prompt injection and jailbreak detection system built with **DistilBERT** and a **Streamlit** web interface. This project is designed to help detect malicious prompts that attempt to manipulate or bypass the behavior of large language models.

## Overview

This project uses a fine-tuned **DistilBERT** model to classify prompts as either:

- **Benign**
- **Prompt Injection / Jailbreak**

It also includes a second layer of filtering for potentially harmful or unsafe content using keyword and pattern-based checks.

## Features

- Prompt injection and jailbreak detection
- Harmful content filtering
- Streamlit-based user interface
- Confidence score and risk level display
- Example prompts for testing
- Fine-tuned DistilBERT model

## Project Files

- `streamlit_app.py` — main Streamlit application
- `prompt_injection_detector.py` — model training and evaluation script
- `prompt_injection_model/` — saved fine-tuned model and tokenizer files
- `requirements.txt` — required Python libraries

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt


## Model Download

The trained model file is too large for standard GitHub upload.

Download the model here:
(https://drive.google.com/drive/folders/1hmgsjTIyXBUxGTTuZzFCZUKul12HvpIU?usp=sharing)

After downloading:
1. Extract the zip file
2. Rename the folder to `prompt_injection_model` if needed
3. Place it beside `streamlit_app.py`

Expected structure:

project-folder/
├── streamlit_app.py
└── prompt_injection_model/
