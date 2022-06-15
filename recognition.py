from speechbrain.pretrained import EncoderDecoderASR
import torchaudio
import torch
import os

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                           savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

# print(asr_model.transcribe_file("enhanced.wav"))

def getAudioName():
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr'
    base2 = r'C:\Users\think\Desktop\temp_SE\snr\testset_txt'
    count = 0
    level = [-8, -4, 0, 4, 8]
    namelist = []
    noises = ['PSTATION', 'SPSQUARE', 'TBUS', 'TCAR', 'TMETRO']
    # print('PSTATION' in 'asd_PSTATIO_N_ad')
    for le in level:
        for noise in noises:
            for file in os.listdir(base + str(le)):
                if noise in file:
                    # print(file)
                    namelist.append(file)
                    break

    list = []
    for i, name in enumerate(namelist):
        for file in os.listdir(base2):
            fs = name.split('_')
            if fs[0] + '_' + fs[1] == file.split('.')[0]:
                with open(os.path.join(base2, file), 'r') as f:
                    content = f.readline().split('.')[0].lower().replace(',', '')
                    list.append((name, content))
    print(list)
    with open('Username.txt', 'w') as f:
        for i in list:
            f.write(i[0] + '|' + i[1] + '\n')
    print(namelist)

def getWord(path):
    y, sr = torchaudio.load(path, channels_first=False)
    waveform = asr_model.audio_normalizer(y, sr)

    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        batch, rel_length
    )
    return predicted_words[0].lower()

if __name__ == '__main__':
    print('Recognition Results:', getWord('./attack/target_cw_enhanced.wav'))