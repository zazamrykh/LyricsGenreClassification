import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer



def get_pretrained(model_name: str, 
                   device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    return tokenizer, model

def parse_model_response(response: str) -> int:
    try:
        match = re.search(r'\{[^}]*"predict"\s*:\s*(0|1)[^}]*\}', response)
        if match:
            data = json.loads(match.group(0))
            return int(data['predict'])
    except Exception as e:
        print(f"Parsing error: {e}")
    raise ValueError("Could not parse prediction from model response.")