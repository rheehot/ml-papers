# PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications.

Tim Slimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

[pixelcnn](https://github.com/openai/pixel-cnn) 은 기존 논문과의 variation이 존재함
1. discretized logistic mixture likelihood
2. condition on whole pixels
3. downsampling to efficiently capture structure at multiple resolutions
4. additional short-cut connections to speed up optimization
5. regularize the model using dropout

## 1. Introduction

pixelcnn은 prob density func on image x를 모든 subpixels에 대해 factorize.

$p(x)=\prod_i p(x_i|x_{<i})$

이 때 conditional prob 은 conv로 모델링. tractable likelihood를 통해 학습함. 아래는 이의 variation.

## 2. Modifications to PixelCNN

### 2.1. Discretized logistic mixture likelihood

std pixelcnn은 모든 subpixel, channel에 conditioning, 256 softmax로 output을 뽑는 방식. 하지만 메모리에 큰 문제를 가졌음. 심지어는 초기 학습에 굉장히 sparse 한 gradient를 생성함. 기존 모델에서는 128이 127과 129와 가까운 값임을 모름. 이걸 low level structure에서 학습해야 higher level로 올라갈 수 있음. 심지어는 어떤 픽셀이 관측되지 않을 때 기본 확률을(=0) 채워넣는 것도 학습해야 함. 이런 경우 observed pixel에서 굉장히 높은 확률이 나왔을 때 치명적. 이런거 막으려고 observed discretized pixel values에 대한 cond prob 게산 방식을 새로 제안하고자 함.

VAE에서 처럼 latent color intensity v가 연속 분포에 존재한다 가정, 이게 8bit로 rounding 되어서 표현되는 것. 이 연속 분포를 mixture of logistics로 가정. 

$\nu \sim \sum^K_{i=1} \pi_i logistic(\mu_i, s_i)\\
P(x|\pi,\mu,s)=\sum^K_{i=1}\pi_i[\sigma ((x + 0.5 - \mu_i)/s_i) - \sigma ((x - 0.5 - \mu_i)/s_i)]$

$\sigma$는 sigmoid, edge case 0은 $-\infty$, 255은 $\infty$로 연산됨. 이 접근은 여러 continuous mixture models에서 발췌, 하지만 0~255 외의 값에 확률을 부여하진 않음. 자연스레 edge value 0, 255에 근방 보다 높은 확률 부여. 이는 실제 통계에 따른 결과. 실험적으로 mixture 수는 5개. 그 결과 더 dense 한 gradient를 얻을 수 있었음. 


### 2.2. Conditioning on whole pixels

## 3. Experiments

## 4. Conclusion

## 5. Citation

1. continuous mixture models
   - Who killed the directed model? : Domke et al., 2018
   - Mixtures of conditional gaussian scale mixtures applied to multiscale image representations: Theis et al., 2012
   - The real-values neural autoregressive density-estimator: Uria et al., 2013
   - Generative image modeling using spatial lstms: Theis & bethge, 2015 