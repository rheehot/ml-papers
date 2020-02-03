# Improving LPCNet-Based Text-To-Speech With Linear Prediction-Structured Mixture Density Network

Min-Jae Hwang, Eunwoo Song, Ryuichi Yamamoto, Frank Soong and Hong-Goo Kang.

## Summary

- lpcnet excitation을 gmm으로 두고, 학습을 원본 시그널과의 nll로 함.
- topic words : lpcnet, mdn, stft loss, dist sharpening
- base model : lpcnet
- variation : 마지막 출력단을 mdn, 학습 과정에서 pred 고려
- benefits : 퀄리티 향상
- weakness : -
- future works : -

## Abstract

이번 논문에서는 LPCNet에 LP-structured mixture density network, LP-MDN을 붙인 Improved LPCNet, iLPCNet을 소개하고자 함. LPCNet은 좋은 성능을 냈지만, 종종 unstable 한데, vocal source가 mu-law 방식으로는 잘 표현되지 않기 때문. 또한 학습 과정이 전체 inference mechanism을 고려하지 않았음. 이러한 문제를 해결하고자 LP-MDN을 소개하고자 하고, 이는 AR vocoder에서 vocal source와 tract 사이의 관계를 표현할 수 있게 함. 

## 1. Introduction

기존까지 wavenet, wavernn, waveglow 등 모델의 성공으로 parametric vocoder를 완전 대체하게 됨. 그중 LPCNet은 source-filter model-based parameteric vocoder과 wavernn의 메리트 모두를 챙긴 모델임. 이 framework에서는 lp-inverse filter로 formant-related spectral structure을 input signal로부터 분리하고, excitation signal을 wavernn으로 확률 분포상 모델링 하는 것. synthesizing에서는 excitation을 만들고 lp synthesis filter로 합성. vocal cord 만으로 excitation이 구성되므로 wavernn 모델링이 더욱 쉬워짐. 반면 unstable한 이유는 mu-law가 충분히 exciation을 표현하지 못하고, production system 전체를 네트워크가 학습하지 못하기 때문.

이를 해결하기 위해 LP-MDN을 제안하고자 함. 이는 학습과 synthesis 과정 모두에 LP structure을 네트워크에 녹일 수 있게 함. 이전 샘플과 lp params가 주어졌다는 가정 하에, speech와 excitation은 constant factor 차이 뿐이었음. 만약 speech가 MOG로 표현되었다면, excitation은 단순히 mean shifting으로 표현될 수 있음. prediction을 excitation MOG에 단순히 더함으로써 speech distribution을 복원하는 것.

1. LP-MDN을 활용해 excitation과 filter 사이의 관계를 capture 함.
2. stft loss나 distribution sharpening 등 효율적인 학습을 추구

## 2. Relationship to prior work

lpc filter를 ar vocoder에 넣으려는 시도는 많았음. 예로 glotnet이나 excitenet은 wavenet 구조로 glottal excitation을 학습하고자 했고, lpcnet은 wavernn을 통해 빠른 ext 생성을 목적으로 한 것.

이와 달리 이 논문의 contribution은 전체 lp synthesis framework를 학습과 synthesis에 모두 넣은 것이고, 이에 속도는 유지하면서 높은 quanlity를 구성할 수 있음.

## 3. LPCNet Vocoder

lpcnet은 lp-based adaptive predictor를 통해 formant structure를 분리함. 그리고 mu-law symbol로 표현된 residual을 wavernn이 학습하는 방식. 

acoustic parameters가 wavernn의 conditional features로 활용됨. 이렇게 생성된 ext를 lp synth filter로 최종 출력을 만들어냄. 

$x_n=e_n+p_n\\ p_n=\sum^M_{i=1}\alpha_ix_{n-i}$

## 4. Proposed Method

LPCNet의 퀄리티 향상을 위해 iLPCNet을 제안하고, 이는 cont dist ext signal을 filter와 함께 학습됨. 먼저 LP-MDN을 소개하고, 이를 합친 iLPCNet 보코더를 소개함. 

### 4.1. Linear prediction-structured mixture density network

LP-MDN을 소개하기 전에 ext와 speech의 prob dist의 관계를 명확히 해야함. $x_n$과 $e_n$의 분포 차는 $p_n$의 상수로 나옴. 
이때 우리가 speech와 ext의 분포를 second-order random var로 가정하면, 다음과 같은 관계로 표현됨.

$p(x_n|\bold x_{<n}, \bold h) = \sum^N_{i=1}w_{n,i}\dot \frac{1}{\sqrt{2\pi}s_{n,i}}\exp\left[ -\frac{(x_n - \mu_{n,i})^2}{2s^2_{n,i}} \right], \\ w^x_{n,i}=w^e_{n,i}, \\ \mu^x_{n,i} = \mu^e_{n,i} + p_n, \\ s^x_{n,i}=s^e_{n,i}$

lp spectral modeling이 mean parameters로 임베딩 되었고, 이는 모델 자체가 ext에 대한 close-loop solution 형태를 띄게 된 것.

$w_n = \mathrm{softmax}(\bold z^w_n), \\ \mu_n = \bold z^\mu_n + p_n, \\ s_n = \exp(\bold z^s_n)$

학습에서는 먼저 MoG (mixture of gaussian)의 likelihood를 계산하고, 이를 토대로 weight을 업데이트 (minimize nll).

### 4.2. Improved LPCNet vocoder

upsampling network로 acoustic features를 이코딩, aux로 활용. waveform generation network 로 시그널 생성. 

### 4.3. Effective training and generation methods

**4.3.1. STFT-based power loss**

aux loss가 perceptual quality에 영향을 미침. 이에 stft power loss 이용

$\mathcal L_{pl} = ||STFT(x) - STFT(\hat x)||^2_2, \\ \mathcal L = \mathcal L_{nll} + \lambda \mathcal L_{pl}$

**4.3.2. Conditional distribution sharpening**

noise가 random sampling 과정에서 발생할 수 있음. 이는 scale param (stddev)로 줄일 수 있음. 그리고 실험적으로 보이스 부분에만 0.7 팩터 정도가 적당했음.

**4.3.3. Training noise injection**

모든 스텝에 mu-law를 뺐다 넣었다 하는건 bottleneck, ilpcnet에서는 linear domain에 gaussian noise 를 추가 (stddev=4/2^16). 이는 2bit error 정도의 수치. 

## 5. Experiments

### 5.1. Experimental setup

**5.1.1. database**

korean female prof speaker 데이터셋, 24k, 16bit, 10 hours.

**5.1.2. Neural vocoders**

ilpcnet, feature net에서는 256dim fc 이용. kernel size, stride interval은 120 (5ms)로 이용. gen net에서는 256, 16 gru, output은 single gaussian으로 가정, 2개 output fc, lambda=10, 모든 fc와 conv 레이어에 weight norm. xavier init, adam opt, noam scheme-based lr scheduling, lr 10e-3, warmup 4k, 10k minibatch per 8gpu, 50 epochs, 530k iter.

## 6. Conclusion

## 7. Citation
