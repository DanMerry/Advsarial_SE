import matplotlib.pyplot as plt
import wave
import numpy as np
import soundfile as sf

def waveform(name):
    # wav = wave.open(name, "rb")
    # params = wav.getparams()
    # print(params)
    # framesra, frameswav = params[2], params[3]  # 获取采样值采样频率
    # datawav = wav.readframes(frameswav)
    # wav.close()
    y, sr = sf.read(name)
    # datause = np.frombuffer(y, dtype=np.short)

    time = np.arange(0, y.shape[0]) * (1.0 / sr)
    plt.ylim([-2, 2])
    plt.plot(time, y)
    plt.show()

def results():
    noise = ['-8', '-4', '0', '4', '8']
    # noise = [1, 2, 3, 4, 5, 6, 7]
    orininal = [56.2867353, 29.99493274, 11.85294535, 12.35196687, 5.590062112]
    original_SE = [45.26107755,	29.7720582,	12.56723106, 12.01928586, 4.161490683]
    orininal_SB = [38.01624172, 25.885343694754, 6.68595252276014, 4.16149068322981, 4.16149068322981]
    cw_SB_100 = [76.6647853503124, 88.4038874235419, 90.8497026763852, 79.5356268375922, 93.5099974474602]
    cw_SB_50 = [78.5922837667924, 80.219631867041, 89.0015031623131, 75.4943135086077, 66.4646189625343]
    cw_SE_100 = [53.26107755, 30.0460308, 12.84120366, 12.84120366, 4.161490683]
    cw_SE_50 = [52.43915974, 33.07495013, 12.56723106, 7.507870331, 4.161490683]
    fgsm_e02 = [44.93209299, 28.97297144, 18.6479007723796, 6.35327150514762, 4.98340849144899]
    pgd_SB = [79.08847859, 73.34172835, 78.43220171, 74.34945215, 65.74201386]
    pgd_SE = [78.89250971, 70.95606795, 73.11057224, 74.37562041, 72.64064967]


    # plt.subplot(1, 2, 1)
    ln1, = plt.plot(noise, orininal, color='red', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln2, = plt.plot(noise, original_SE, color='orange', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln3, = plt.plot(noise, orininal_SB, color='blue', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln4, = plt.plot(noise, cw_SB_100, color='lightseagreen', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln5, = plt.plot(noise, cw_SE_100, color='yellow', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln6, = plt.plot(noise, pgd_SB, color='black', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    ln7, = plt.plot(noise, pgd_SE, color='tab:blue', linewidth=2.0, linestyle='--', marker='o', markersize=5)

    plt.legend(handles=[ln1, ln2, ln3, ln4, ln5, ln6, ln7],
               labels=['Original example', 'MetricGAN+ enhanced', 'SEGAN enhanced', 'OPT on MetricGAN+', 'OPT on SEGAN', 'PGD on MetricGAN+', 'PGD on SEGAN'])
    plt.xlabel('SNR')
    plt.ylabel("WER(%)")
    plt.ylim(0, 100)
    plt.grid()

    # plt.subplot(1, 2, 2)
    # ln4, = plt.plot(noise, cw_SB_100, color='red', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    # ln5, = plt.plot(noise, cw_SE_100, color='orange', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    # ln6, = plt.plot(noise, fgsm_e02, color='blue', linewidth=2.0, linestyle='--', marker='o', markersize=5)
    # plt.legend(handles=[ln4, ln5, ln6],
    #            labels=['OPT on MetricGAN+', 'OPT on SEGAN', 'FGSM on MetricGAN+'])
    # plt.xlabel('SNR')
    # plt.title("WER(%)")
    # plt.ylim(0, 100)
    # plt.grid()

    # 设置标题及字体
    # plt.legend(handles=[ln1, ln2, ln3, ln4, ln5, ln6], labels=['ordinary noise on TIMIT', 'ordinary noise on LibriSpeech', 'ordinary noise on CommonVoice', 'Gaussian noise on TIMIT', 'Gaussian noise on LibriSpeech', 'Gaussian noise on CommonVoice'])
    plt.savefig('wer.png')
    plt.show()

def rectangle():

    similarity = [0.651, 0.802, 0.86]  # similarity of action
    divergence = [0.21748494048186232, 0.012685957834863208, 0.0027021711873171966]  # js diversity

    fgsm_RPR = [0.328588063, 0.476921863, 0.531077203, 0.615891087, 0.590024567]
    fgsm_DE = [6.915851272, 3.087627745, 11.96194825, 2.191780822, 0.821917808]
    cw_RPR = [1.605821259, 1.443004601, 1.303557011, 1.251898765, 1.254088374]
    cw_DE = [38.64854363, 62.51854373, 84.16375015, 75.37413615, 89.34850676]


    labels = ['Random PPO', 'Replicating model', 'Baseline']
    labels = [-8, -4, 0, 4, 8]

    # plt.rcParams['axes.labelsize'] = 16  # xy轴label的size
    # plt.rcParams['xtick.labelsize'] = 12  # x轴ticks的size
    # plt.rcParams['ytick.labelsize'] = 14  # y轴ticks的size
    # plt.rcParams['legend.fontsize'] = 12  # 图例的size

    # 设置柱形的间隔
    width = 0.2  # 柱形的宽度
    x1_list = []
    x2_list = []
    x3_list = []
    x4_list = []
    for i in range(len(fgsm_RPR)):
        x1_list.append(i)
        x2_list.append(i + width)
        x3_list.append(i + width * 2)
        x4_list.append(i + width * 3)

    # 创建图层
    fig, ax1 = plt.subplots()
    shape = r'-'

    # 设置左侧Y轴对应的figure
    ax1.set_ylabel('Residual Perturbation Rate')
    ax1.set_xlabel('SNR')
    ax1.set_ylim(0, 2.0)
    ln1 = ax1.bar(x1_list, fgsm_RPR, width=width, color=(255, 0, 0), align='edge', label='sda')
    ln2 =ax1.bar(x2_list, cw_RPR, width=width, color='lightseagreen', align='edge')


    # ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴

    # 设置右侧Y轴对应的figure
    ax2 = ax1.twinx()
    ax2.set_ylabel('Degree of Enhancement')
    # ax2.set_ylim(0, 0.5)
    ln3 = ax2.bar(x3_list, fgsm_DE, width=width, color='blue', align='edge', tick_label=labels)
    ln4 = ax2.bar(x4_list, cw_DE, width=width, color='tab:blue', align='edge', tick_label=labels)

    plt.xlabel('SNR')
    plt.legend(handles=[ln1, ln2, ln3, ln4], bbox_to_anchor=(0.5, 1),
               labels=['RPR on FGSM', 'RPR on OPT', 'DE on FGSM', 'DE on OPT'])

    plt.tight_layout()
    plt.savefig("DEandRPR.png")
    plt.show()


if __name__ == '__main__':
    results()
    #waveform('attack/adv_cw_new_enhanced.wav')