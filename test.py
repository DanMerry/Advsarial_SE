import torch
import soundfile as sf
import torchaudio
import speechbrain
from torchaudio import _extension
from speechbrain.pretrained import SpectralMaskEnhancement
import pyAudioAnalysis.ShortTermFeatures
from torchsummary import summary

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="./pretrained_models/metricgan-plus-voicebank",
    savedir="./pretrained_models/metricgan-plus-voicebank",
)
model = enhance_model.hparams.enhance_model
# Load and add fake batch dimension
noisy = enhance_model.load_audio(
    "./pretrained_models/metricgan-plus-voicebank/reback.wav",
    savedir="./pretrained_models/metricgan-plus-voicebank"
).unsqueeze(0)
print('noisy', noisy.shape)

# Add relative length tensor
enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

# myself load  the data
input = enhance_model.compute_features(noisy)
print('input', input.shape)
output_1 = model(input, lengths=torch.tensor([1.]))
output_2 = torch.mul(output_1, input)
output_last = enhance_model.hparams.resynth(torch.expm1(output_2), noisy)
print('output:', output_last.shape)
print(enhanced.shape)
# y, sr = sf.read("./pretrained_models/metricgan-plus-voicebank/reback.wav")
# signal, sr2 = torchaudio.load("./pretrained_models/metricgan-plus-voicebank/reback.wav")
# print(y)
# print(signal, sr2)

# Saving enhanced signal on disk

torchaudio.save('reback_denoise.wav', output_last.cpu(), 16000)
y, sr = sf.read('reback_denoise.wav')
print(y.shape)
print(model)

