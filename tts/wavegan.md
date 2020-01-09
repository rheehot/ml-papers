# WaveGAN: Adversarial Audio Synthesis

Chris Donahue, Julian McAuley, Miller Puckette.

## Summary

- DCGAN을 통해 1초 audio patch 생성 실험.
- topic words : DCGAN, Phase shuffle, WGAN-GP
- base model : DCGAN
- variation : 2D conv를 오디오 특성에 맞게 1D로 변경
- benefits : unconditional audio generation에 성공함
- weakness : 음질이 아직 멀었음
- future works : 음질 향상이 필요

## Abstract

오디오 시그널은 high temporal resolution에서 샘플링 되고, range of timescales 에서 스트럭쳐를 캡쳐하는 방식을 학습해야 한다. 여기서 소개하고자 하는건 WaveGAN이고, raw-waveform audio를 생성하는 첫번째 GAN 모델이 될 것. 

## 1. Introduction

GAN은 오디오 분야에 꽤 많이 보임. data-hungry speech recog system에서 data augmentator로써 작용하기도 하고 (Shrivastaa et al., 2017), 한번에 다량을 single-forward만으로 sampling할 수 있음. 하지만 fidelity가 증가하면서 GAN은 아직 unsupervised setting에서 오디오를 생성할 수 있는지 가능성을 보이지 않음.

가장 쉬운 방법은 spectrogram을 이미지처럼 다루는 것. 이건 discriminative setting에서 종종 있는 일. 하지만 generative setting에서 spectrogram은 invertible 하지 않으므로 griffin-lim이나 vocoder 없이 들을 수 없음. 

이번에는 one-sec slice 오디오를 gan을 통해 만들어 볼거. spectrogram approach를 specgan으로 명명, DCGAN을 통해 모델링. waveform approach를 wavegan, dcgan을 1D로 re-represent. 

이번 연구의 목표는 unsupervised strategies로 global structure를 학습할 수 있는지, high-dimensional audio signal에서도 conditioning 없이 작동하는지임. 

## 2. GAN Preliminaries

Generator G: Z -> X와 Discriminator D: X -> {0, 1}에 대해서 two-player minimax game을 진행.

$V(D, G) = \mathbb E_{x \sim P_X} [\log D(x)] + \mathbb E_{z \sim P_Z}[\log (1 - D(G(z)))]$

그리고 이게 JS div를 minimize 하는 global optima를 찾는 방법론으로써 작용. 이게 학습하기도 어렵고 해서 WGAN에서는 wasserstein-1 distance를 사용

$W(P_X, P_G) = \sup_{||f||_L \le 1} \mathbb E_{x \sim P_X}[f(x)] - \mathbb E_{x \sim P_G}[f(x)]$

여기서 $||f||_L \le 1 : X -> \mathbb R$은 1-Lipschitz 군의 함수.

$V_{WGAN}(D_w, G) = \mathbb E_{x \sim P_X}[D_w(x)] - \mathbb E_{x \sim P_G}[D_w(G(z))]$

weight clipping이나 gradient penalty를 통해 D_w가 1-Lipschitz이게 하고, D_w는 학습 보다는 wasserstein distance를 계산하는 역할을 함. 그리고 여기는 WGAN-GP (Gradient Penalty)를 이용.

## 3. WaveGAN

### 3.1. Intrinsic differences between audio and images

가장 먼저 어떤 axis에 어떤 데이터가 분포했는지 확인 ex. PCA. 이미지는 주로 intensity, gradient, edge를 잡는다면 오디오에서는 periodic basis, 주기성을 보이는 경향이 있음.

또한 waveform이 16k 등 resolution이 높은만큼 larger receptive field를 요구. 

### 3.2. WaveGAN Architecture

DCGAN을 활용하였고, transposed conv를 변경하여 receptive field를 넓히는데 활용. 2D 5x5 필터 대신 1D 25-length 필터를 사용하였고, upsample 에서는 2배 대신 4배를 이용함. 이를 통해 같은 크기의 parameters와 비슷한 수의 operations를 구성함.

DCGAN이 64x64 이미지를 만들면 실상 오디오는 4096 길이인데, 여기에 layer 하나를 더 얹어 16384 길이의 오디오를 만듬. 이는 16k 오디오의 1초 길이. future work 로 megapixel image generation technique를 통해 분단위 오디오를 만들어 볼 것. 

requant를 통해 16비트 데이터를 32bit floating point로 변환, generator도 32비트 실수로 오디오를 뽑을거. normalization 없애고, wgan-gp로 학습.

### 3.3. Phase Shuffle

checkerboard artifacts를 없애야 하는데, 이미지에서는 이 패턴이 흔하지 않아서 discriminator가 쉽게 걷어낼 수 있는데, 오디오에서는 periodicity로 인해 오브젝티브 설정이 더 어려워짐. 만약 이게 특정 phase에서만 발생하여 disc가 trivial policy를 학습한다면 문제가 생길 수 있음.

이거 막으려고 하는게 phase shuffle. disc에서 phase를 Uniform(-n, n) 샘플한 수만큼 회전. disc가 학습에 phase만을 관찰하는 것을 막고 싶은 것.

## 4. SpecGAN: Generating Semi-Invertible Spectrograms

freq domain 생서 모델. spectrogram representation을 만들건데 근사적으로나마 invertible 해야함. 16k sample에 128x128 스펙트로그램을 만들거임.

STFT 하고 16ms window, 8ms stride, 128 freq bin으로 만들고, amplitude는 log를 맥임. 그리고 N(0, 1)이 되도록 normalize 하고 [-1, 1]로 rescale.

## 5. Experimental Protocol

zero부터 nine까지 읽는 데이터셋에서 돌리기. 남음 실험 조건은 논문 보기. wavegan은 200k에 4일, specgan은 1750 epoch에 이틀.

## 6. Evaluation Methodology

## 7. Results and Discussion

WaveGAN에서는 dropout은 넣을 떄 더 별로였고, phase shuffle은 Uniform(-2, 2)일 때, SpecGAN은 phase shuffle이 더 벌로였고, griffin-lim 말고 다른 보코더를 썼으면 더 좋았을 지도. 

## 8. Related Work

 ## 9. Conclusion


