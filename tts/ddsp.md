# DDSP: Differentiable Digital Signal processing

Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, Adam Roberts

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

생성 모델은 시간이나 주파수 도메인에서 샘플을 만들어 내는데, 시그널을 나타내는데는 적절하겠지만, 그를 표현하는데는 음성을 생성하고 전달하는 existing knowledge를 전혀 반영하지 않는다는 점에서 비효율적인 표현. vocoder, synthesizer approach는 성공적으로 signal processing의 domain knowledge를 적용했지만, auto-differentiation-based ML method를 이용해서 이를 표현하는데 어려워 연구가 많지 않다. 

이 논문에서 소개하고자 하는건 DDSP로 이는 classic signal processing elements를 DL 메소드와 합치는데 도움을 준다. audio synth에서 high-fidelity generation에 성공했는데 이에 large ar model이나 adv loss 없이도 nn의 expressive power 손실 없이 학습할 수 있었음. 또한 interpretable module를 통해 model component를 명시적으로 manipluate 할 수 있게 함. 

DDSP는 modular approach를 통해 interpretable하고 generative 하지만 dl의 이점을 놓치지 않음.

## 1. Introduction

NN은 asymptotic limit 내에서의 universal approximator이다. 이의 practical success는 structural priors인 convolution, reccurence, self-attention에 의해서이다. 이러한 architectural constriants는 일반화와 data efficiency에 대해 data domain과 맞물려 좋은 결과를 보임. 이러한 관점에서 end2end learning이 structural prior과 규모에 의존적인데 실험자는 dfferentiability에 의해 툴 박스 자체에 제한이 생김. 이러한 tool box를 늘이려는 시도가 DDSP, TF 기반으로 작동한다. broad applicability를 가지고, potential에 대해서 앞으로 이야기할 것.

objects는 주기적으로 진동하는 natural tendency를 가진다. 작은 형태의 변위는 스프링과 같은 에너지를 보존하는 탄성력에 의해 복원되는데, 이는 운동 에너지와 위치 에너지 사이의 harmonic oscillation에 의해 작동한다. 이에 따라 인간의 청각은 phase-coherent oscillation에 민감하게 진화했으며, basilar membrane과 청각 피질(auditory cortex)로의 tonotopic mapping의 울림의 속성을 통해 오디오를 spectrotemporal responses로 분리한다. 하지만 neural synthesis 모델은 generation과 perception에 periodic structure를 주로 이용하지 않는다.

### 1.1. Challenges of neural audio synthesis

대부분의 neural synth 모델은 time domain에서 시그널을 생성하거나 frequency domain에서의 fourier coeff를 추론하는 식으로 시그널을 생성한다. 이러한 표현은 어떠한 형태의 waveform도 생성할 수 있지만, bias에서 자유롭지 않다. 이러한 이유는 oscillations 대신에 aligned wave packets를 오디오 생성 과정에 prior로써 적용하기 때문이다. 예로 strided convolution models는 frame을 overlapping하면서 waveform을 직접생성한다. audio가 다양한 프리퀀시에서 oscillate하기 때문에 모델은 두 프레임 사이의 waveform을 정확히 정렬해야 하고, 모든 phase variation을 커버칠 수 있는 filter를 학습해야 한다.

태코트론 같은 Fourier-based models는 phase alignment problem을 가진다. 또한 sinusoids와 같은 spectral leakage를 완벽히 복원할 수 있어야 한다. 

WaveNet 같은 AR 모델은 single sample씩 만들면서 앞선 이슈를 피할 수 있다. 하지만 data-hungry networks를 요구로 하고, oscillation이라는 bias를 충분히 활용하지 못한다. teacher-forcing을 이용하면 생성 중에 exposure bias가 발생하고, error가 복잡해질 수 있다. 이는 perceptual loss, pretrained model, discriminator 등과의 접속을 어렵게 한다. 또한 학습 자체가 비효율적인데 이는 waveform의 shape이 달라도 perception이 동일 할 수 있기 때문.

## 2. Related work

## 3. DDSP Components

## 4. Experiments

## 5. Results

## 6. Conclusion

## 7. References
