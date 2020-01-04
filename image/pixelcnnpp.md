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

기존 pixelcnn은 3개 subpixel에 대해서도 factorize. 근데 이게 모델을 사실 어렵게 하는 일이었음. 새 모델에서는 feature maps를 R/G/B scale의 정보 공유 여부에 따라 3개 그룹으로 나눔. 먼저 red를 context에서 뽑고, 이를 기반으로 green과 blue를 연쇄적으로 뽑는 방식. 

$p(r_{i, j},g_{i, j},b_{i, j}|C_{i, j})=P(r_{i, j}|\mu_r(C_{i, j}),s_r(C_{i, j}))\times P(g_{i, j}|\mu_g(C_{i, j}), s_g(C_{i, j})) \times P(b_{i, j}|\mu_b(C_{i, j},r_{i, j},g_{i, j}),s_b(C_{i, j}))\\
\mu_g(C_{i, j},r_{i, j})=\mu_g(C_{i, j})+\alpha (C_{i, j})r_{i, j}\\
\mu_b(C_{i, j},r_{i, j},g_{i, j})=\mu_b(C_{i, j})+\beta(C_{i, j})r_{i, j}+\gamma (C_{i, j})b_{i, j}$

### 2.3. Downsampling versus dilated convolution

명시적으로 이미지 quality를 위해 long-term dep를 만들어주기 위해 dilated conv를 활용. 이 때 dilation으로 인한 정보 손실을 방지하기 위해 additional short-cut connections를 제공.

### 2.4. Adding short-cut connections

U-net처럼 동일 resolution feature map에 대해 추가 residual connection 제공

### 2.5. Regularization using dropout

overfitting 된 모델이 생성한 이미지가 perceptuality가 떨어지는걸 실험적으로 관측, regularizing을 위해 residual path 에 std bin dropout을 추가. 

## 3. Experiments

### 3.3. Examining Network depth and field of view size

명시적인 receptive field가 작았을 때 더 좋은 성능을 CIFAR-10에서 보였었음. receptive field를 줄이면서 network capacity (expressiveness)를 늘이기 위해서 두가지 방법을 제안함. 
1. NIN(Network in network): gated resnset block with 1x1 conv를 추가. 
2. autoregressive channel: 1x1 conv gated resnetblock를 통해 channels간 skip connection 추가. 


### 3.4. Ablation experiments

**3.4.1. Softmax likelihood instead of discretized logistic mixture**

softmax likelihood가 더 flexible하긴 했음. 하지만 학습 속도랑 inference 속도가 느리긴 하더라. 

**3.4.2. Continuous mixture likelihood instead of discretization**

입력 이미지를 dequantize 하고 uniform noise 를 첨가함으로써 continuous로 모델링 할 수도 있음. 그럼 결과적으로 vae 프레임워크를 따르는데, dequantized z가 latent, 모델에 의해 capture된 prior를 따를 것. entropy는 uniform dist가 0이니까 dequantized pixel의 log likelihood만 남음. 결과 2.92 비트 per dim으로 굉장히 잘 학습함. 

**3.4.3. No short-cut connection**

학습의 하한을 낮춤 

**3.4.4. No dropout**

test set에 대한 loglikelihood를 굉ㅈ아히 많이 낮춤. 

## 4. Conclusion

잘나왔음 끝~

## 5. Citation

1. continuous mixture models
   - Who killed the directed model? : Domke et al., 2018
   - Mixtures of conditional gaussian scale mixtures applied to multiscale image representations: Theis et al., 2012
   - The real-values neural autoregressive density-estimator: Uria et al., 2013
   - Generative image modeling using spatial lstms: Theis & bethge, 2015 