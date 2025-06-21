import streamlit as st
import torch
from newspaper import Article
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F


class BertANNClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, d1=256, d2=128, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()
        hidden_size = self.bert.config.hidden_size
        self.fc1 = nn.Linear(hidden_size, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.out = nn.Linear(d2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# Load your fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("my_finetuned_bert_tk")
    model = BertANNClassifier(model_name='bert-base-cased', num_labels=3)
    model.load_state_dict(torch.load(r"C:\Users\mudas\Downloads\my_models\my_finetuned_bert_ANN", map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

st.title("News Article Political Bias Identifier")

url = st.text_input("Enter the URL of a news article:")

def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error extracting article: {str(e)}")
        return None

# Map class indices to labels
class_labels = {0: "Left", 1: "Center", 2: "Right"}

if st.button("Predict Bias"):
    if url:
        text = get_article_text(url)
        if text:
            st.write("Article Extracted:")
            st.write(text[:500] + "..." if len(text) > 500 else text)
            # Preprocess and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                predicted_class = torch.argmax(outputs, dim=1).item()
            # Use the label mapping
            label = class_labels.get(predicted_class, "Unknown")
            st.success(f"Predicted Bias: {label}")
        else:
            st.error("Could not extract article text. Please check the URL.")
    else:
        st.warning("Please enter a URL.")
