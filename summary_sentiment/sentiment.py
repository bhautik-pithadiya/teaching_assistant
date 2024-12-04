# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_sentiment_model():

    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions",ignore_mismatched_sizes=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model.to(device)
    return tokenizer,model

def inference(text,tokenizer,model):
    
    # print(type(text))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    return model.config.id2label[int((predicted_class_ids)[0])]
