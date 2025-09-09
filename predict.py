import torch
import torchaudio
import json
import os
import torch.nn as nn


# --- 1. Copy model definition from your training script ---
# This script needs to know what your model looks like, so we copy the AudioTransformer class code
class AudioTransformer(nn.Module):
    def __init__(self, num_input_features=128, num_classes=35, dropout=0.1):
        super().__init__()
        # Using PyTorch's pre-built Transformer components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_input_features, nhead=4, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(num_input_features, num_classes)

    def forward(self, spectrogram_batch):
        # Input shape needs to be (batch, time, features) for batch_first=True
        # Spectrograms are often (batch, features, time), so we might need to permute
        x = spectrogram_batch.permute(0, 2, 1)

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over the time dimension
        predictions = self.output_layer(x)
        return predictions


# --- 2. Define inference function ---
def predict(model, audio_path, label_map_reverse, device):
    """
    Load a WAV file, perform prediction and return results
    """
    # a. Set model to evaluation mode
    model.eval()

    # b. Prepare data transformation pipeline (must be exactly the same as during training)
    transform = torchaudio.transforms.MelSpectrogram(n_mels=128)
    target_length = 81

    # c. Load and transform audio
    waveform, _ = torchaudio.load(audio_path)
    spectrogram = transform(waveform)

    # d. Pad or truncate to target length (must be exactly the same as during training)
    current_length = spectrogram.shape[2]
    if current_length < target_length:
        padding_needed = target_length - current_length
        spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_needed))
    elif current_length > target_length:
        spectrogram = spectrogram[:, :, :target_length]

    # e. Prepare input tensor for model
    #    - .unsqueeze(0) adds a dimension at the front to simulate a batch
    # input_tensor = spectrogram.unsqueeze(0).to(device)
    # New line (correct)
    input_tensor = spectrogram.to(device)

    # f. Perform prediction (in no_grad context to save computational resources)
    with torch.no_grad():
        predictions = model(input_tensor)

    # g. Interpret output
    #    - Softmax converts model output to probabilities
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    #    - Argmax finds the index of the class with highest probability
    predicted_index = torch.argmax(probabilities).item()
    #    - Use label_map_reverse to find corresponding word
    predicted_label = label_map_reverse[predicted_index]

    return predicted_label, probabilities[predicted_index].item()


# --- 3. Main program ---
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "<your model path>"  # Path to your saved best model
    # !! Modify this to the WAV file path you want to test !!
    WAV_FILE_PATH = "<the wav file in SpeechCommands folder you want to test>"

    # !! The label_map_reverse here must be exactly the same as in your training script !!
    # You can copy from the output of your training script, or load a saved label_map file
    full_data_path = os.path.join(
        os.path.dirname(__file__), "/data/SpeechCommands/speech_commands_v0.02"
    )
    labels = []
    for item in os.listdir(full_data_path):
        item_path = os.path.join(full_data_path, item)
        if os.path.isdir(item_path) and not item.startswith("_") and item != "LICENSE":
            labels.append(item)

    # Sort labels for consistent mapping
    labels.sort()
    # For convenience, I'll manually create a simplified one here
    all_labels = labels
    all_labels.sort()  # Ensure order is the same as during training
    print(all_labels)
    label_map_reverse = {idx: label for idx, label in enumerate(all_labels)}

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First, create a model instance with exactly the same structure as when saved
    model = AudioTransformer(num_classes=len(label_map_reverse)).to(device)

    # Then, load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print(f"Model loaded from {MODEL_PATH}")

    # --- Run prediction ---
    predicted_word, confidence = predict(
        model, WAV_FILE_PATH, label_map_reverse, device
    )

    # --- Display results ---
    print("-" * 30)
    print(f"Audio File: {WAV_FILE_PATH}")
    print(f"Predicted Word: '{predicted_word}'")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 30)
