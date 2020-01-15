# Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-Resolution Spectrogram.

Ryuichi Yamamoto, Eunwoo Song and Jae-Min Kim

## Summary

- WaveNet + GAN 모델에 multi-res STFT 로스
- topic words : WaveNet, GAN, Multi-resolution STFT
- base model : WaveNet
- variation : GAN, multi-res STFT
- benefits : realistic, fast, no-distilation 모델 구현.
- weakness : GAN의 고질적인 음성 artifact 존재.
- future works : 이게 noise 때문인지 아닌지도 확인해 봐야 할 듯 함. 

## Abstract

Parallel WaveGAN을 제안하고 GAN을 통해 distillation-free, fast, small-footprint한 PWG를 구현함. non-autoregressive WaveNet은 multi-resolution spectrogram을 GAN을 통해 오디오로 만듬.

Parallel WaveGAN은 density distillation을 요구하지 않음. 파라미터 적고, 속도도 빠름. 

## 1. Introduction

WaveNet은 기존 vocoder에 비해서 월등한 성능을 보였지만, 인퍼런스 속도가 너무 느리다는 autoregressive model의 본질적 한계를 가졌음.

이번 논문은 PWG에서 PDD 과정에 일어나는 teacher-student framework의 고질적 한계를 타파하고자 함. PDD 에서 teacher-student는 좋은 teacher 뿐만 아니라 추가적인 error methodology가 필요함 (perceptual loss 등).

이를 해결하고자 만든게 parallel wavegan인데, GAN을 통한 PWG 진행. 기존과 달리 teacher-student의 two-stage가 아닌 싱글 스테이지 모델. 오직 non-autoregressive WaveNet을 multi-resolution STFT와 adversarial loss를 통해 학습한 것. 

1. multi-resolution STFT 랑 time-domain adversarial loss를 활용. 

2. teacher-student가 필요하지 않음. simple-pass 이므로 학습 속도도 빠르고, 인퍼런스 속도도 빨라졌음.

3. Transformer 기반의 TTS에 보코더를 Parallel WaveGAN을 쓰니까 더 좋았음.

## 2. Related work

기존에 teacher-student 기반 WaveNet PWG PDD 모델에 kld와 aux, adv loss를 더한 연구가 있었음. percetual quality는 올랐지만, 복잡한 학습 단계와 유연성을 줄이는 단점을 가졌음. 

이러한 단점을 보완하기 위해 two-stage를 single-stage로 바꾸기 위한게 목표였고, 이미 GELP에서 비슷한 시도가 있었음. 하지만, TTS acoustic model에서 오류가 생겼을 때 발생하는 LPC 오류로 인한 에러는 잡을 수 없었음. 

그래서 여기서는 LPC 없이 곧바로 웨이브 폼을 만드려고 함. 이 때 speech signal의 유동성은 상당히 학습이 어려운데 이를 해결하기 위해서 multi-resolution STFT loss를 사용하고자 함. 

## 3. Method

### 3.1. Parallel waveform generation based on GAN

WaveNet based generator를 사용하는데, mel을 condition으로 하고 non-causal conv를 통해 noise를 non-autoregressive하게 오디오로 생성함. 

$L_{adv}(G, D) = \mathbb E_{z \sim N(0, I)}[(1 - D(G(z)))^2]\\
L_D(G, D) = \mathbb E_{x \sim P_{data}}[(1 - D(x))^2] + \mathbb E_{z \sim N(0, I)}[D(G(z))^2]$

### 3.2. Multi-resoluton STFT auxiliary loss

stability와 efficiency를 위해 multi-resolution STFT를 auxiliary loss로 제안함. 

$L_s(G) = \mathbb E_{z \sim p(z), x\sim p_{data}}[L_{sc}(x, \hat x) + L_{mag}(x, \hat x)]$

$L_sc$는 spectral convergence loss, $L_{mag}$는 log STFT magnitude loss.

$L_{sc}(x, \hat x) = \frac{||\ |STFT(x)| - |STFT(\hat x)| \ ||_F}{||\ |STFT(x)| \ ||}\\
L_{mag}(x, \hat x) = \frac{1}{N}||\ \log|STFT(x)| - \log|STFT(\hat x)| \ ||_1$

frobenius 와 $L_1$ norm. |STFT()|는 STFT magnitude. 

Multi-resolution STFT는 STFT의 파라미터가 다른거. M을 STFT 로스의 수라고 하면, aux loss는

$L_{aux}(G) = \frac{1}{M}\sum^M_{m=1}L^{(m)}_s(G)$

이 때 resolution 마다 time과 frequency level의 trade-off가 존재하는데, 예로 window size가 증가하면 higher frequency resolution을 표현하지만, temporal resolution이 줄어듬. 이렇게 해서 여러개의 STFT loss를 더해 둘 다 챙기려는 생각. 만약 STFT가 몇개 안되면서 생기는 오버피팅 문제도 해결할 수 있었음.

$L_G(G, D) = L_{aux}(G) + \lambda_{adv}L_{adv}(G, D)$

위를 최종 generator loss로 산정. 

## 4. Experiments

wavenet은 3dilated cycle, non-causal, dilated 30 layer, 3 kernel size. disc는 10 x dilated non-causal conv. 전체에 weightnorm. 

STFT는 3개 loss. disc는 각 step 별로 산정한 확률의 평균. $L_{adv}$는 4. 400k, radam 1e-6. 

aux 피쳐는 knn으로 upsampling, 2D conv 패싱. 

## 5. Conclusion

distillation-free, fast, small-footprint generation에 성공했고, realistic했음. 

## 6. Acknowledgements

## 6. References
