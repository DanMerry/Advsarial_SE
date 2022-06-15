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

def accoringToSNR(snr):
    base = './attack/data/original_enhanced'
    files = os.listdir(base)
    words = []
    for file in files:
        fs = file.split('snr')
        if fs[-1].split('_')[0] == str(snr):
            print(file)
            word = getWord(os.path.join(base, file))
            print(word)
            words.append((file, word))
    with open('./text/original_enhanced/snr' + str(snr) + '.txt', 'w') as f:
        for ws in words:
            f.write(ws[0] + '|' + ws[-1] + '\n')
    print('Done!!')

def accoringToSNRADV(snr):
    base = './attack/data/cw_enhanced'
    files = os.listdir(base)
    words = []
    for file in files:
        fs = file.split('snr')
        if fs[-1].split('_')[0] == str(snr) and fs[-1].split('_')[1] == '100':
            print(file)
            word = getWord(os.path.join(base, file))
            print(word)
            words.append((file, word))
    with open('./text/cw_enhanced/snr' + str(snr) + '_100.txt', 'w') as f:
        for ws in words:
            f.write(ws[0] + '|' + ws[-1] + '\n')
    print('Done!!')

def calWER(snr):
    results = []
    with open('text/fgsm_enhanced/snr' + str(snr) + '_e0.2.txt', 'r') as f:
        lines = f.readlines()
    with open('Username.txt', 'r') as f2:
        grounds = f2.readlines()

    for line in lines:
        # print(line.split('|')[0].split('_enhanced')[0] + '.wav')
        for ground in grounds:
            if line.split('|')[0].split('_e0.2_enhanced')[0] + '.wav' == ground.split('|')[0]:
                gt = ground.split('|')[-1].replace('\n', '')
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
    with open('./text/cw_enhanced/snr' + str(snr) + '_100.txt', 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            adv_enhanced.append(line.split('|')[0])

    originals = []
    for adv in adv_enhanced:
        originals.append(adv.split('_100_enhanced')[0] + '.wav')

    original_enhanceds = []
    for original in originals:
        for file in os.listdir('attack/data/SB/original_enhanced'):
            if file.split('_enhanced')[0] + '.wav' == original:
                original_enhanceds.append(file)
    advs = []
    for original in originals:
        for file in os.listdir('attack/data/SB/cw_example'):
            if file.split('_100')[0] + '.wav' == original:
                advs.append(file)

    # print(adv_enhanced)
    # print(originals)
    # print(original_enhanceds)
    # print(advs)

    rpr = []
    for i in range(len(originals)):
        adv = os.path.join('attack/data/SB/cw_example', advs[i])
        x_adv, sr = sf.read(adv)

        adv_en = os.path.join('attack/data/SB/cw_enhanced', adv_enhanced[i])
        y_adv, sr = sf.read(adv_en)

        ori = os.path.join(base, originals[i])
        x, sr = sf.read(ori)

        ori_en = os.path.join('attack/data/SB/original_enhanced', original_enhanceds[i])
        y, sr = sf.read(ori_en)

        res = np.log(np.linalg.norm(y - y_adv)) / np.log(np.linalg.norm(x - x_adv))
        rpr.append(res)
    print(rpr)
    print('Avg:', np.mean(rpr))
    print()

def metric_RPR_fgsm(snr):
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr' + str(snr)

    adv_enhanced = []
    with open('./text/fgsm_enhanced/snr' + str(snr) + '_e0.2.txt', 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            adv_enhanced.append(line.split('|')[0])

    originals = []
    for adv in adv_enhanced:
        originals.append(adv.split('_e0.2')[0] + '.wav')

    original_enhanceds = []
    for original in originals:
        for file in os.listdir('attack/data/SB/original_enhanced'):
            if file.split('_enhanced')[0] + '.wav' == original:
                original_enhanceds.append(file)
    advs = []
    for original in originals:
        for file in os.listdir('attack/data/SB/fgsm_example'):
            if file.split('_e0.2')[0] + '.wav' == original:
                advs.append(file)

    # print(adv_enhanced)
    # print(originals)
    # print(original_enhanceds)
    # print(advs)

    rpr = []
    for i in range(len(originals)):
        adv = os.path.join('attack/data/SB/fgsm_example', advs[i])
        x_adv, sr = sf.read(adv)

        adv_en = os.path.join('attack/data/SB/fgsm_enhanced', adv_enhanced[i])
        y_adv, sr = sf.read(adv_en)

        ori = os.path.join(base, originals[i])
        x, sr = sf.read(ori)

        ori_en = os.path.join('attack/data/SB/original_enhanced', original_enhanceds[i])
        y, sr = sf.read(ori_en)

        res = np.log(np.linalg.norm(y - y_adv)) / np.log(np.linalg.norm(x - x_adv))
        rpr.append(res)
    print(rpr)
    print('Avg:', np.mean(rpr))
    print()

def metric_DE_cw(snr):
    with open('Username.txt') as f:
        lines = f.readlines()

    with open('./text/cw_enhanced/snr' + str(snr) + '_100.txt', 'r') as f2:
        adv_en_contexts = f2.readlines()

    groungtrth = []
    for adv_en_context in adv_en_contexts:
        for line in lines:
            if adv_en_context.split('|')[0].split('_100_enhanced')[0] + '.wav' == line.split('|')[0]:
                groungtrth.append(line)

    with open('./text/original_enhanced/snr' + str(snr) + '.txt', 'r') as f2:
        en_orig = f2.readlines()

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

def metric_DE_fgsm(snr):
    with open('Username.txt') as f:
        lines = f.readlines()

    with open('./text/fgsm_enhanced/snr' + str(snr) + '_e0.2.txt', 'r') as f2:
        adv_en_contexts = f2.readlines()

    groungtrth = []
    for adv_en_context in adv_en_contexts:
        for line in lines:
            if adv_en_context.split('|')[0].split('_e0.2_enhanced')[0] + '.wav' == line.split('|')[0]:
                groungtrth.append(line)

    with open('./text/original_enhanced/snr' + str(snr) + '.txt', 'r') as f2:
        en_orig = f2.readlines()

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

def getOriginaltxt(snr):
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr' + str(snr)
    with open('Username.txt', 'r') as f2:
        grounds = f2.readlines()
    num = 0
    results = []
    pres = []
    for ground in grounds:
        if ground.split('|')[0].split('.wav')[0].split('snr')[-1] == str(snr):
            name = os.path.join(base, ground.split('|')[0])
            pre = getWord(name)
            gt = ground.split('|')[-1].replace('\n', '')
            pres.append((ground.split('|')[0], gt))
            num += 1
            results.append(wer.wer(gt, pre))
    with open(os.path.join('text/SB/original', 'snr' + str(snr) + '.txt'), 'w') as f:
        for pre in pres:
            f.write(pre[0] + '|' + pre[1] + '\n')
    print('num:{}, res:{}'.format(num, results))
    return np.mean(results)

def metric_RPR_pgd(snr):
    base = r'C:\Users\think\Desktop\temp_SE\snr\snr' + str(snr)

    adv_enhanced = []
    with open('./text/SB/pgd_enhanced/snr' + str(snr) + '_s50e0.002.txt', 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            adv_enhanced.append(line.split('|')[0])

    originals = []
    for adv in adv_enhanced:
        originals.append(adv.split('_s50e0.002')[0] + '.wav')

    original_enhanceds = []
    for original in originals:
        for file in os.listdir('attack/data/SB/original_enhanced'):
            if file.split('_enhanced')[0] + '.wav' == original:
                original_enhanceds.append(file)
    advs = []
    for original in originals:
        for file in os.listdir('attack/data/SB/pgd_example'):
            if file.split('_s50e0.002')[0] + '.wav' == original:
                advs.append(file)

    # print(adv_enhanced)
    # print(originals)
    # print(original_enhanceds)
    # print(advs)

    rpr = []
    for i in range(len(originals)):
        adv = os.path.join('attack/data/SB/pgd_example', advs[i])
        x_adv, sr = sf.read(adv)

        adv_en = os.path.join('attack/data/SB/pgd_enhanced', adv_enhanced[i])
        y_adv, sr = sf.read(adv_en)

        ori = os.path.join(base, originals[i])
        x, sr = sf.read(ori)

        ori_en = os.path.join('attack/data/SB/original_enhanced', original_enhanceds[i])
        y, sr = sf.read(ori_en)

        res = np.log(np.linalg.norm(y - y_adv)) / np.log(np.linalg.norm(x - x_adv))
        rpr.append(res)
    print(rpr)
    print('Avg:', np.mean(rpr))
    print()
    return np.mean(rpr)

if __name__ == '__main__':
    SNR = [-8, -4, 0, 4, 8]
    avg = []
    # for i in SNR:
    #     print('SNR:', i)
    #     avg.append(calWER(i))
    # print(avg)

    # for i in SNR:
    #     print('SNR:', i)
    #     accoringToSNRADV(i)

    # Calculate RPR
    # for i in SNR:
    #     avg.append(metric_RPR_pgd(i))
    # print(avg)
    # print(np.mean(avg))

    # Calculate DE
    # avgs = []
    # for i in SNR:
    #     avgs.append(metric_DE_fgsm(i))
    # print(avgs)

    # avgs = []
    # for i in SNR:
    #     avgs.append(getOriginaltxt(i))
    # print(avgs)
    base = r'attack/data/SB/cw_target_enhanced'
    files = os.listdir(base)
    # for i in range(25):
    #     print(files[i])
    #     File = os.path.join(base, files[i])
    #     print(getWord(File))
    print(getWord(os.path.join(base, 'p257_258_TMETRO_0_snr0_1000_enhanced.wav')))

'''
what was the name
there is a new difference
what was the main difference
what is the object of it
what is the name dear prince
what are the good reparts
not that he gave reference
what was the main difference
what was the name indecorous
edward together with the others
orders made with her
then i came in with her
'''