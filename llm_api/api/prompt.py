import os
import json

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")

import os

def get_prompt(frase):
    prompt_filename = f"prompt.txt"
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_filename)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_filename}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    return base_prompt.replace("{{phrase}}", frase)

def get_prompt_rule(frase):
    prompt_filename = f"prompt_rules.txt"
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_filename)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_filename}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    return base_prompt.replace("{{phrase}}", frase)

def get_prompt_frase(frase):
    prompt_filename = f"prompt_frase.txt"
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_filename)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_filename}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    return base_prompt.replace("{{phrase}}", frase)