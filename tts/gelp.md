# GELP: GAN-Excited Linear Prediction for Speech Synthesis from Mel-Spectrogram

Lauri Juvela, Bajibabu Bollepalli, Junichi Yamagishi, Paavo Alku.

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

audio2mel은 잘 나왔는데 vocoder라는 새로운 challenge. wavenet 같은 neural vocoder가 있지만, autoregressive model로 인해 성능이 느림. parallel inference는 학습도 어렵고 모델 크기도 커지는 문제가 있음. 그래서 이번에는 GAN과 linear predictive synthesis filter를 이용하기로 함. 그 결과 더 빠르고 성능 좋은 모델을 만들 수 있었음.

## 1. Introduction

Neural vocoder 까지의 발전에서 WaveNet은 high quality 오디오를 생성하는데 큰 기여를 했지만, sample-by-sample sequential inference 방식에 의해 느린 inference 속도를 가졌음. RNN을 optimizing 하거나 dilation buffering, parallel-application에 의해서 많이 해결하고자 했음. parallel-app의 경우 teacher-student 모델을 이용하였음.

그리고 여기에는 STFT aux loss 가 많이 이용됨. Source-filter과 LP 쪽에서는 LP로 모델링한 시그널에 대해 excitation을 neural network로 학습하는 방식. 이게 flow와 접목되기도 했었음. 하지만 이또한 여전히 sequential한 operation을 함. 

mfcc에서 유도된 all-pole envelopes에 대해 gan을 기반으로 residual excitation modeling을 한 연구도 있음. 하지만 pitch info에 굉장히 디펜던트한 문제가 있었음. 

TTS 입장에서 mel 만으로는 excplicit voicing이나 pitch 정보가 없기 때문에 challenging한 문제. 그리고 이건 waveform reconstruction으로 돌아가긴 함. 

## 2. Methods

3개 모델로 구성. Generator, discriminator, conditionner. 모든 모델은 non-causal 1d conv를 이용. conditionner은 context embedding c를 mel로 부터 frame rate에 따라 만들고, linear하게 upsample 해서 audio rate에 맞춰짐. 그리고 이건 G와 D에 들어갈 공통 정보. generator에는 z를 넣고, lp residual를 모델링. lp synthesis filter에 넣어서 결과를 뽑아냄. 

voicing이나 pitch 없이도 충분히 mel로 만들 수 있었음. G랑 C 학습할 때는 stft magnitude 관련 로스랑 time domain adv loss. 

### 2.1. Envelope recovery from mel-spectrogram

mel-spectrogram은 STFT magnitude spectrogram에 mel-filterbank랑 log를 써서 넣는건데, 복원을 위해서 M의 pseudo inverse를 취하고, negative value를 막기 위해 다음과 같이 복원.

$\hat X = \max(M^+ \exp(m), \epsilon)$

이후 IFFT거쳐서 auto-correlation 연산, LPC를 추출함. 

### 2.2. Synthesis filter for parallel inference

LP Polynomial에 zero padding해서 FFT를 돌린다 가정.

$A_k = FFT{a_k}$

sign-inverted phase와 magnitude를 통해 synthesis filter 구성.

$H_k = \frac{\exp(-id\angle A_k)}{\max(|A_k|, \epsilon)}$

이를 통해 frequency level에서 prediction을 해서 istft로 복원. 전부 differentiable함.

$\hat x = ISTFT\{STFT\{\hat e\} \odot H\}$

### 2.3. Network Architectures

dilated conv residual network에 gated activation + skip connection 포함 (WaveNet과 흡사). 대신 skip connection은 1x1 conv대신 affine transform을 붙여 사용.

G랑 C는 timestep 맞추려고 zero padding 쓰고, D는 residual connection을 하지 않음. causal 하지 않기 때문에 bidirectional한 형태임.

### 2.4. Losses

WGAN 사용.

$\mathcal L_{GAN} = - \mathbb E_{x \sim p_x} [D(x, c)] + \mathbb E_{\hat x \sim p_G}[D(\hat x, c)]$

Disc를 충분히 smooth하게 두기 위해 gradient penalty 부여.

$L_{GP} = \mathbb E_{x \sim p_x, \hat x \sim p_G}[(||\nabla_{\hat x} D(\tilde x, c)|| - 1)^2]$

where $\tilde x = \epsilon x + (1 - \epsilon ) \hat x$. 또한 gradient magnitude도 regularize. 이건 real data sample과 dynamics에 대한 convergence를 위해.

$\mathcal L_{R1} = \mathbb E_{x \sim p_x}[||\nabla_x D(x, c||^2]$

마지막은 STFT loss.

$\mathcal L_{STFT} = \mathbb E[(|STFT\{x\}| - |STFT\{\hat x\})^2]$

total objective.

$\mathcal L_{G, C} = \lambda_1 \mathcal L_{STFT} - \mathcal L_{GAN}\\
\mathcal L_D = \mathcal L_{GAN} + \lambda_2 \mathcal L_{GP} + \lambda_3 \mathcal L_{R1}$

## 3. Experiments

$\lambda$는 순서대로 10, 10, 1. residual excitation domain에서 프리트레인도 진행. 

## 4. Discussion

## 5. Conclusions

GELP를 만들었고, 잘 나옴.

## 6. Citation
- Neural source filter-based waveform model for statistical parammeteric speech synthesis
- Speech waveform synthesis from mfcc sequences with generative adversarial networks