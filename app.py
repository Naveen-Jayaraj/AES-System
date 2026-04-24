import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, Batch
from flask import Flask, request, jsonify, render_template
import torch.nn.functional as F
import numpy as np

# --- 1. Define the Model Architecture ---
# This class must be identical to the one used for training.
class GraphBertAES(nn.Module):
    def __init__(self, model_name, embedding_dim, graph_dim, hidden_dim):
        super(GraphBertAES, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.graph_fc = nn.Linear(graph_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, graph):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        graph_emb = F.relu(self.graph_fc(graph.x.mean(dim=0, keepdim=True)))
        combined = torch.cat([bert_out, graph_emb.repeat(bert_out.size(0), 1)], dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x).squeeze(-1)

# --- 2. Configuration and Model Loading ---
MODEL_NAME = "roberta-base"
EMBEDDING_DIM = 768
GRAPH_DIM = 64
HIDDEN_DIM = 128
MODEL_SAVE_PATH = "graphbert_aes_model_final.pth"
MIN_SCORE = 1
MAX_SCORE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Initialize the model and load the trained weights
model = GraphBertAES(MODEL_NAME, EMBEDDING_DIM, GRAPH_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()  # Set the model to evaluation mode

print("✅ Model loaded successfully!")

# --- 3. Flask Web Application ---
app = Flask(__name__)

# Function to predict the score for a given essay
def predict_score(essay_text):
    # Tokenize the input essay
    enc = tokenizer(
        essay_text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # Move tensors to the correct device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Create a dummy graph (as used in training)
    # The model expects a graph, so we provide one.
    dummy_graph = Data(x=torch.randn(10, GRAPH_DIM), edge_index=torch.randint(0, 10, (2, 20)))
    graph_batch = Batch.from_data_list([dummy_graph]).to(device)

    # Get the prediction
    with torch.no_grad():
        output = model(input_ids, attention_mask, graph_batch)
        predicted_score = output.item()
    
    # Clip the score to be within the valid range [1, 5]
    clipped_score = np.clip(predicted_score, MIN_SCORE, MAX_SCORE)
    
    # Round to one decimal place for a cleaner output
    return round(clipped_score, 1)

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the essay scoring API request
@app.route('/score-essay', methods=['POST'])
def score_essay():
    data = request.get_json()
    essay = data.get('essay', '')
    
    if not essay.strip():
        return jsonify({'error': 'Essay text cannot be empty.'}), 400
        
    try:
        score = predict_score(essay)
        return jsonify({'score': score})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred while scoring the essay.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
