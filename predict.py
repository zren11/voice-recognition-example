import torch
import torchaudio
import json
import os
import torch.nn as nn


# --- 1. 从你的训练脚本中复制模型定义 ---
# 这个脚本需要知道你的模型长什么样，所以我们把 AudioTransformer 类的代码复制过来
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


# --- 2. 定义推理函数 ---
def predict(model, audio_path, label_map_reverse, device):
    """
    加载一个WAV文件，进行预测并返回结果
    """
    # a. 将模型设置为评估模式
    model.eval()

    # b. 准备数据转换流程 (必须和训练时完全一样)
    transform = torchaudio.transforms.MelSpectrogram(n_mels=128)
    target_length = 81

    # c. 加载和转换音频
    waveform, _ = torchaudio.load(audio_path)
    spectrogram = transform(waveform)

    # d. 填充或截断至目标长度 (必须和训练时完全一样)
    current_length = spectrogram.shape[2]
    if current_length < target_length:
        padding_needed = target_length - current_length
        spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_needed))
    elif current_length > target_length:
        spectrogram = spectrogram[:, :, :target_length]

    # e. 准备输入模型的张量
    #    - .unsqueeze(0) 在最前面增加一个维度，模拟一个批次 (batch)
    # input_tensor = spectrogram.unsqueeze(0).to(device)
    # New line (correct)
    input_tensor = spectrogram.to(device)

    # f. 进行预测 (在 no_grad 环境下，节省计算资源)
    with torch.no_grad():
        predictions = model(input_tensor)

    # g. 解读输出
    #    - Softmax 将模型的输出转换成概率
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    #    - Argmax 找到概率最高的那个类别的索引
    predicted_index = torch.argmax(probabilities).item()
    #    - 使用 label_map_reverse 找到对应的单词
    predicted_label = label_map_reverse[predicted_index]

    return predicted_label, probabilities[predicted_index].item()


# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- 配置 ---
    MODEL_PATH = "best_model_20250906-215258.pth"  # 你保存的最佳模型的路径
    # !! 修改这里为你想要测试的 WAV 文件路径 !!
    WAV_FILE_PATH = "SpeechCommands/speech_commands_v0.02/eight/0a196374_nohash_0.wav"

    # !! 这里的 label_map_reverse 必须和你的训练脚本中的完全一致 !!
    # 你可以从训练脚本的输出中复制，或者加载保存的 label_map 文件
    full_data_path = os.path.join(
        os.path.dirname(__file__), "SpeechCommands/speech_commands_v0.02"
    )
    labels = []
    for item in os.listdir(full_data_path):
        item_path = os.path.join(full_data_path, item)
        if os.path.isdir(item_path) and not item.startswith("_") and item != "LICENSE":
            labels.append(item)

    # Sort labels for consistent mapping
    labels.sort()
    # 为了方便，我这里先手动创建一个简化的
    all_labels = labels
    all_labels.sort()  # 确保顺序和训练时一样
    print(all_labels)
    label_map_reverse = {idx: label for idx, label in enumerate(all_labels)}

    # --- 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 首先，创建一个和保存时结构完全相同的模型实例
    model = AudioTransformer(num_classes=len(label_map_reverse)).to(device)

    # 然后，加载保存的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print(f"Model loaded from {MODEL_PATH}")

    # --- 运行预测 ---
    predicted_word, confidence = predict(
        model, WAV_FILE_PATH, label_map_reverse, device
    )

    # --- 显示结果 ---
    print("-" * 30)
    print(f"Audio File: {WAV_FILE_PATH}")
    print(f"Predicted Word: '{predicted_word}'")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 30)
