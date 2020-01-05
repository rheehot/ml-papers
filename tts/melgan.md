# MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis

Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

GAN을 통해서 high quality coherent waveforms를 간단한 training framework에서 학습할 수 있는 것을 보일것임. mel inversion에 효과적이었음. ablation studies에서 모델의 컴포넌트를 뜯어볼 것. non-autoregressive 하고 fully-convolutional 이며 parameters 수가 적음. 

## 1. Introduction

raw audio를 모델링하는 것이 어려운 이유는 1초에 16kHz 샘플을 뽑는만큼 temporal resolution이 높은 편. 또한 short-term, long-term 등 서로 달느 timescale에서 structure dep이 존재함. 그래서 더 쉽게 연산하려고 lower-resolution으로 줄여서 연산한거. 이게 mel 같은거. 이건 inversion을 할 수 있을만큼 충분한 정보를 담고 있는 repr이어야 함. 

주로 두가지 단계로 모델링이 들어가는데 text2mel, mel2audio 이런 느낌. 이 때 mel 대신에 bark-cepstrum 같은 다른 intermediate를 쓰기도 함. 

모델은 크게 3개로 봄.

**Pure signal processing approaches** 

griffin-lim 같은 algorithmic한 방법론. stft sequence를 다시 signal로 복원하는 메소드. 더 나아가서는 WORLD vocoder 에서는 mel-like intermediate를 소개하고, dedicated signal과 함께 원본 시그널로 복원하는 역할을 함. Char2Wav의 보코더로도 채용됨.

**Autoregressive neural-networks-based models**

WaveNet 은 fully-convolutional autoregressive sequence model. 하지만, receptive field의 한계로 몇 초 이상의 dependency는 어려움. 이걸 해결하기 위한게 SampleRNN. 얘는 multi-scale에서 different resolution에 대해 모델링. WaveRNN은 single recurrent entwork, 여러 테크닉을 통해 가속화한 모델. 전체적으로 추론 단계에서 느리고 비효율적인 샘플링 방식을 가짐. 

**Non autoregressive models**

autoregressive한 시퀸스 사이의 dependency가 없기 때문에 parallelizable하고, deep learning hardware 사용량을 늘릴 수 있음.

1. Parallel WaveNet and Clarinet

이건 autoregressive 방식으로 train한 모델을 flow-based convolutional student model에 nn distillation한 것. 기본적으로 KL을 쓰고 부가적인 perceptual loss를 가짐

2. WaveGlow

Glow와 같은 flow based model을 이용. 근데 8개 GPU로 1주일이 걸릴만큼 학습이 느림. 이게 autoregressive 방식인 AF는 inf 단계에서 autoreg가 발생하고, train에서 parallel하게 작동하고, IAF는 inf에서 parallel하게 작동하고, train에서 autoreg가 발생. 그래서 이걸 싹 정리한게 WaveFlow. 

**GANs for audio**

GAN이 vision에서 큰 영향을 미친 것과 별개로, audio 분야에서는 드믐. Yamamoto는 autoregressive 모델에서 distillation을 GAN을 통해서 진행했는데, adversarial loss 만으로는 부족했고, KL 기반의 distillation objective가 필요했음.

**Main Contributions**

- MelGAN이란 모델을 소개할거고, non-autoregressive feed-forward convolutional arch이며 GAN 베이스로 동작. GAN으로 distillation 이나 perceptual loss 없이 학습한 첫 작
- parallel MelGAN decoder 로 autoregressive 모델이 대체될 것.
- MelGAN이 제일 빠름.

## 2. The MelGAn Model

![figure1](./rsrc/melgan_fig1.png)

### 2.1. Generator

**Architecture**

generator은 mel s 를 raw waveform x 를 뽑느 fully convolutional feed-forward network. 기본 x256 lower resol mel을 가정. upsampling layer은 transposed conv, 뒤에 dilated residual conv blocks. 

기존 GAN과 달리 noise를 이용하지 않음. noise를 추가했을 때 perceptual difference가 존재했음. 이게 실제 기존 Mathieu et al., 2015, Isola et al., 2017의 논문에서 봤든 conditioning info가 굉장히 강한 경우 noise vector가 큰 영향을 끼치지 못했음. 

**Induced Receptive Field**



## 3. Results

## 4. Conclusion and future works

## 5. Citation
- WORLD vocoder: MORISE et al., 2016.
- SampleRNN: Mehriet al., 2016.
- Clarinet: Ping et al., 2018.
- Probability density distillation with generative adversarial networks for high-quality parallel waveform generation: Yamamoto et al., 2019.