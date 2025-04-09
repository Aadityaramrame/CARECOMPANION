import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator

class MedicalSummary:
    def __init__(self, model_path='/Users/aditi/Desktop/CareCompanion/fine_tuned_model'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = self.load_model()

    def load_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model

    def summarize_text(self, text, max_length=250):
        try:
            input_ids = self.tokenizer.encode(f'summarize: {text}', return_tensors='pt').to(self.device)
            summary_ids = self.model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Summarization failed: {e}")
            return None
