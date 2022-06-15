import librosa
import soundfile as sf

y, sr = librosa.load('./pretrained_models/metricgan-plus-voicebank/example.wav', sr=16000)
y_8k = librosa.resample(y=y, orig_sr=16000, target_sr=8000)
sf.write('sample_8k.wav', data=y_8k, samplerate=8000)
