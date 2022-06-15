import os
import wer
import torch
import numpy as np
import torchaudio
import soundfile as sf
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                           savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

def getWord(path):
    y, sr = torchaudio.load(path, channels_first=False)
    waveform = asr_model.audio_normalizer(y, sr)

    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        batch, rel_length
    )
    return predicted_words[0].lower()

# It's used for all files, needed to be modified a little!
def writeInTxt(snr):
    base = r'./attack/data/SB/cw_target_enhanced'
    files = os.listdir(base)
    pres = []
    for file in files:
        if file.split('_')[-3].split('snr')[-1] == str(snr) and file.split('_')[-2] == '1000':
            File = os.path.join(base, file)
            pres.append((file, getWord(File)))
    print(pres)
    with open('./text/SB/target_attack/snr' + str(snr) + '_1000.txt', 'w') as f:
        for pre in pres:
            f.write(pre[0] + '|' + pre[1] + '\n')
    return

def calWER(snr):
    results = []
    with open('text/SB/target_attack/snr' + str(snr) + '_1000.txt', 'r') as f:
        lines = f.readlines()
    with open('Username.txt', 'r') as f2:
        grounds = f2.readlines()

    for line in lines:
        # print(line.split('|')[0].split('_enhanced')[0] + '.wav')
        for ground in grounds:
            if line.split('|')[0].split('_1000_')[0] == ground.split('.wav')[0] and line.split('|')[0].split('_')[-2] == '1000':
                # gt = ground.split('|')[-1].replace('\n', '')
                gt = 'what was the main difference'
                pre = line.split('|')[-1].replace('\n', '')
                # print('ground_truth:', gt)
                # print('predicted:', pre)
                results.append(wer.wer(gt, pre))
    print(results)
    print('Avg:', np.mean(results))
    return np.mean(results)

def metric_RPR_cw(snr):
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr' + str(snr)

    adv_enhanced = []
    with open('./text/SB/afromSE/cw_enhanced/snr' + str(snr) + '_100.txt', 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            adv_enhanced.append(line.split('|')[0])

    originals = []
    for adv in adv_enhanced:
        originals.append(adv.split('_100_enhanced')[0] + '.wav')

    original_enhanceds = []
    for original in originals:
        for file in os.listdir('./attack/data/SE/original_enhanced'):
            if file.split('_enhanced')[0] + '.wav' == original:
                original_enhanceds.append(file)
    advs = []
    for original in originals:
        for file in os.listdir('./attack/data/SE/cw_example'):
            if file.split('_100')[0] + '.wav' == original:
                advs.append(file)

    # print(adv_enhanced)
    # print(originals)
    # print(original_enhanceds)
    # print(advs)

    rpr = []
    for i in range(len(originals)):
        adv = os.path.join('attack/data/SE/cw_example', advs[i])
        x_adv, sr = sf.read(adv)

        adv_en = os.path.join('attack/data/SB/fromSE/cw_enhanced', adv_enhanced[i])
        y_adv, sr = sf.read(adv_en)
        #y_adv = np.concatenate((y_adv, x_adv[y_adv.shape[0]:]))

        ori = os.path.join(base, originals[i])
        x, sr = sf.read(ori)

        ori_en = os.path.join('attack/data/SE/original_enhanced', original_enhanceds[i])
        y, sr = sf.read(ori_en)
        # print(y.shape)
        y = np.concatenate((y, y_adv[y.shape[0]:]))
        # print(y.shape)

        # print(y.shape, y_adv.shape, x.shape, x_adv.shape)

        res = np.log(np.linalg.norm(y - y_adv)) / np.log(np.linalg.norm(x - x_adv))
        rpr.append(res)
    print(rpr)
    print('Avg:', np.mean(rpr))
    print()
    return np.mean(rpr)

def metric_DE_cw(snr):
    with open('Username.txt') as f:
        lines = f.readlines()

    with open('./text/SE/afromSB/pgd_enhanced/snr' + str(snr) + '_s50e0.002.txt', 'r') as f2:
        adv_en_contexts = f2.readlines()

    groungtrth = []
    for adv_en_context in adv_en_contexts:
        for line in lines:
            if adv_en_context.split('|')[0].split('_s50e0.002')[0] + '.wav' == line.split('|')[0]:
                groungtrth.append(line)

    with open('./text/SB/original_enhanced/snr' + str(snr) + '.txt', 'r') as f3:
        en_orig = f3.readlines()

    de = []
    for i in range(len(en_orig)):
        print(groungtrth[i].split('|')[0])
        print(adv_en_contexts[i].split('|')[0])
        print(en_orig[i].split('|')[0])
        print()

        gt = groungtrth[i].split('|')[-1].replace('\n', '')
        adv = adv_en_contexts[i].split('|')[-1].replace('\n', '')
        ori = en_orig[i].split('|')[-1].replace('\n', '')

        de.append(wer.wer(gt, adv)-wer.wer(gt, ori))

    print(de)
    print('AVG:', np.mean(de))
    return np.mean(de)

def metric_RPR_pgd(snr):
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr' + str(snr)

    adv_enhanced = []
    with open('./text/SB/afromSE/pgd_enhanced/snr' + str(snr) + '_s50e0.004.txt', 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            adv_enhanced.append(line.split('|')[0])

    originals = []
    for adv in adv_enhanced:
        originals.append(adv.split('_s50e0.004')[0] + '.wav')

    original_enhanceds = []
    for original in originals:
        for file in os.listdir('./attack/data/SB/original_enhanced'):
            if file.split('_enhanced')[0] + '.wav' == original:
                original_enhanceds.append(file)
    advs = []
    for original in originals:
        for file in os.listdir('./attack/data/SE/pgd_example'):
            if file.split('_s50e0.004')[0] + '.wav' == original:
                advs.append(file)

    # print(adv_enhanced)
    # print(originals)
    # print(original_enhanceds)
    # print(advs)

    rpr = []
    for i in range(len(originals)):
        adv = os.path.join('attack/data/SE/pgd_example', advs[i])
        x_adv, sr = sf.read(adv)

        adv_en = os.path.join('attack/data/SB/fromSE/pgd_enhanced', adv_enhanced[i])
        y_adv, sr = sf.read(adv_en)
        #y_adv = np.concatenate((y_adv, x_adv[y_adv.shape[0]:]))

        ori = os.path.join(base, originals[i])
        x, sr = sf.read(ori)

        ori_en = os.path.join('attack/data/SB/original_enhanced', original_enhanceds[i])
        y, sr = sf.read(ori_en)
        # print(y.shape)
        # y = np.concatenate((y, y_adv[y.shape[0]:]))
        # print(y.shape)

        print(y.shape, y_adv.shape, x.shape, x_adv.shape)

        res = np.log(np.linalg.norm(y[:y_adv.shape[0],] - y_adv)) / np.log(np.linalg.norm(x[:x_adv.shape[0],] - x_adv))
        # res = np.log(np.linalg.norm(y - y_adv)) / np.log(np.linalg.norm(x[:x_adv.shape[0],] - x_adv))
        rpr.append(res)
    print(rpr)
    print('Avg:', np.mean(rpr))
    print()
    return np.mean(rpr)

def metric_DE_pgd(snr):
    with open('Username.txt') as f:
        lines = f.readlines()

    with open('./text/SB/afromSE/pgd_enhanced/snr' + str(snr) + '_s50e0.004.txt', 'r') as f2:
        adv_en_contexts = f2.readlines()

    groungtrth = []
    for adv_en_context in adv_en_contexts:
        for line in lines:
            if adv_en_context.split('|')[0].split('_s50e0.004_enhanced')[0] + '.wav' == line.split('|')[0]:
                groungtrth.append(line)

    with open('./text/SE/original_enhanced/snr' + str(snr) + '.txt', 'r') as f3:
        en_orig = f3.readlines()

    de = []
    for i in range(len(en_orig)):
        print(groungtrth[i].split('|')[0])
        print(adv_en_contexts[i].split('|')[0])
        print(en_orig[i].split('|')[0])
        print()

        gt = groungtrth[i].split('|')[-1].replace('\n', '')
        adv = adv_en_contexts[i].split('|')[-1].replace('\n', '')
        ori = en_orig[i].split('|')[-1].replace('\n', '')

        de.append(wer.wer(gt, adv)-wer.wer(gt, ori))

    print(de)
    print('AVG:', np.mean(de))
    return np.mean(de)

if __name__ == '__main__':
    SNR = [-8, -4, 0, 4, 8]
    avg = []

    # original examples
    for snr in SNR:
        avg.append(calWER(snr))
    print(avg)
    print(np.mean(avg))

    # original examples
    # for snr in SNR:
    #     writeInTxt(snr)

    # RPR
    # for snr in SNR:
    #     avg.append(metric_RPR_pgd(snr))
    # print(avg)
    # print(np.mean(avg))

    # DE
    # for snr in SNR:
    #     avg.append(metric_DE_pgd(snr))
    # print(avg)
    # print(np.mean(np.abs(avg)))

    # write in
    # for snr in SNR:
    #     writeInTxt(snr)