# Probability Density Distillation with Generative Adversarial Networks for High-Quality Parallel Waveform Generation

Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

WaveNet-based parallel waveform generation (PWG)에서 probability density distillation (pdd)를 잘하는 방법론을 소개. teacher-student framework가 성공적이었지만, auxiliary losses 가 없으면 퀄리티가 많이 떨어졌음. 이를 보완하기 위해 GAN 기반 프레임워크를 제안함. IAF 기반 studnet를 gen으로 가정, PDD 학습을 adversarial mechanism을 따름. 이를 통해 더 natural한 결과를 얻을 수 있었음. 

## 1. Introduction

wavenet은 autoregressive structure 때문에 그 inference 속도가 너무 느렸음. 그래서 나온게 teacher-student framework였고, 이를 통해 parallel 하게 결과를 뽑을 수 있었음. e.g., parallel wavenet, clarinet. 이는 주로 pdd를 통해 IAF 기반 모델을 feed-forward 만으로 학습시키는 방식. 

PDD는 주로 KL을 줄이는 방식으로 학습. 단 KLD 만으로는 부족해서 auxiliary loss를 부가적으로 이용 했음. 하지만 수렴이 어려워 unexpected artifacts를 만들기도 했음.

이번에 제안하고자 하는건 하나의 generalized optimization criteria. 그리고 이는 IAF와 GAN 구조를 차용한 것. WaveNet은 mle로 학습하고, IAF를 gan의 gen으로 가정, adversarial method를 통해 opt. gan을 통해 더 realistic한 분포를 학습하기 때문에 naturalness가 증가함. 또한 joint opt with conventional distillation은 long-term dep를 가진 모델을 GAN으로 학습하는데 어려움을 겪었는데, 이것도 효율적으로 해결했음.

## 2. Related work

parallel wavenet은 kld 를 objective로 뒀고, clarinet은 여기에 frame level stft loss를 추가함. stft loss는 iaf student가 time-frequency의 스트럭쳐도 학습하게 함.

GAN이 오디오 synthesis에서도 잘 작동했음. 저자의 목적은 teacher-student framework에 adversarial training을 이식해서 더 좋은 결과를 얻는 것. 기존의 연구에서는 distillation이 목적이 아니라, 이미 학습된 student에 speaker adaptation을 진행했지만 성공하지 못했음. 

제안하고자 하는 criteria는 KLD + STFT loss + GAN based adversarial framework. 결과 더 좋은 성능을 가지게 되었다. 

## 3. Probability density distillation

### 3.1. KLD distillation

기본적으로 KLD를 가지고 trained WaveNet을 IAF에 정보 전달을 목표. IAF는 inf를 Parallel하게 돌릴 수 있으므로 훨씬 빠른 성능을 보임.

먼저 IAF는 z noise를 x로 변환하고, x는 teacher에 의해 evaluating 됨. 이후 KLD를 통해 두 output 분포를 비교하는 방식.

$L_{KLD}(p, q)=\mathbb E_{z, \hat x} [\sum^T_{t=1}KL^{reg}(q(\hat x_t|z_{<t})||p(\hat x_t|\hat x_{<t}))]$

### 3.2. STFT-based auxiliary loss

KLD 만으로는 부족해서 auxiliary loss를 추가. 이번 페이퍼에서는 frame-level auxiliary loss를 기용. 

$L_{AUX}(q)=\mathbb E_{x, \hat x}[L_{SC}(x, \hat x) + \lambda_{MAG}L_{MAG}(x, \hat x)]$

$x$는 타겟, $\hat x$는 estimated. $\lambda_{MAG}$는 두 로스의 비를 조절하고, spectral convergence loss($L_{SC}$) 와 log STFT magnitude loss($L_{MAG}$)를 의미. 

$L_{SC}(x, \hat x) = \frac{||\ |STFT(x)| - |STFT(\hat x)|\ ||_F}{||\ |STFT(x)|\ ||_F}\\
L_{MAG}(x, \hat x)=||\ log|STFT(x)| - log|STFT(\hat x)|\ ||_1$

$||\ ||_F$는 frobenius norm. $|STFT(\cdot)|$는 STFT의 magnitude. spectral convergence loss는 spectral peaks를 강조하고, log STFT magnitude loss는 spectral valleys를 맞누는 역할을 함.

## 4. Probability density distillation with generative adversarial networks

$L_G(q, p, D)=\lambda_{kld}L_{KLD}(q, p) + \lambda_{aux}L_{AUX}(q) + \lambda_{adv}L_{ADV}(q, D)\\
L_{ADV}(q, D) = \mathbb E_{\hat x \sim q}[(1 - D(\hat x))^2]$

generator loss는 kld, aux, adverarial loss를 모두 합친 형태. 

$L_D(q, D) = \mathbb E_{x \sim p_{data}}[(1 - D(x))^2] + \mathbb E_{\hat x \sim q}[D(\hat x)^2]$

## 5. Experiments

## 6. Conclusion

WaveNet-based PWG의 학습 과정에 PDD + GAN framework를 이용함. 그 결과 더 좋은 음성을 얻을 수 있었음. 

## 7. Acknowledgements

## 8. Citation

