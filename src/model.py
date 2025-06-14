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

def predict_genre():
    pass