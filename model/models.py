import torch
import torch.nn as nn
# Load model directly
from transformers import AutoModelForSeq2SeqLM


class CVEClassifier(nn.Module):
    """
    A BiLSTM based model for CVE patch commit prediction.
    """
    def __init__(self, lstm_hidden_size, num_classes, lstm_layers=1, dropout=0.1):
        super().__init__()
        # Initialize the pre-trained model for embeddings
        self.codeReviewr = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
        # LSTM parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.codeReviewr.config.hidden_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output
        self.fc = nn.Linear(self.lstm_hidden_size*2, num_classes)  # times 2 for bi-LSTM

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg, attention_mask_msg, input_ids_diff, attention_mask_diff):
        # Getting embeddings for all inputs
        desc_embed = self.codeReviewr(input_ids=input_ids_desc, attention_mask=attention_mask_desc).last_hidden_state
        msg_embed = self.codeReviewr(input_ids=input_ids_msg, attention_mask=attention_mask_msg).last_hidden_state
        diff_embed = self.codeReviewr(input_ids=input_ids_diff, attention_mask=attention_mask_diff).last_hidden_state
        # Concatenate along the sequence length dimension (dim=1)
        concatenated = torch.cat((desc_embed, msg_embed, diff_embed), dim=1)
        # Pass the concatenated embeddings to the LSTM
        lstm_output, _ = self.lstm(concatenated)
        # Apply dropout
        dropped = self.dropout_layer(lstm_output)
        # Pass through the fully connected layer to get the output
        output = self.fc(dropped[:, -1, :])  # using only the last hidden state
        return output
