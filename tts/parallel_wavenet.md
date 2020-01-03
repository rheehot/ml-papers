# Parallel WaveNet: Fast High-Fidelity Speech Synthesis

Aaron van den Oord, Yazhe Li, Igor Babuschkin, Karen Simonyan, Oriol Vinyals, Koray Kavukcuoglu.

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

WaveNet이 Sequential 한 생성 단계 때문에 parallel computers에 잘 안 맞고 real-time에 안 뽑힘.
=> Probability density distillation을 이용해서 parallel 하게 wavenet을 돌리는 것이 목적.

## 1. Introduction

WaveNet이 SOTA를 이루긴 했지만, 리얼 월드에서는 생성 속도의 한계에 부딫힘. 이를 distilling 해서 feed-forward하게 바꿔봄.

현재는 24kHz의 sample rate로도 돌아가니 autoregression이 어려움. 그래서 영감을 얻은게 IAF (WaveFlow 같은거). WaveNet이랑 IAF의 장점만 모으기 위해 neural network distillation이 필요했고, 이를 probability density distillation 이라 하기로 함. 그리고 이게 feedforward IAF의 teacher로 활용될 것.

## 2. WaveNet

$p(x)=\prod_tP(x_t|x_{<t},\theta)$

과거 시퀸스 전체를 조건화해서 output을 만듬. 전형적인 autoregressive.

$p(x_t|x_{<t})$를 만들기 위해 한번의 forward pass가 요구됨. 이 때 쓰이는게 causal dilated conv. 근데 이게 학슴 때는 모든 정보를 가지고 있으니까 gpu 수준에서 parallel 하게 연산이 가능한데, synthesis 에서는 unit 하나 당 forward pass가 요구되니 굉장히 느림.

현대는 16~24kHz 정도의 sample rate를 가지기 때문에 이런 signal의 long-term dep을 모델링 하기 위해선 네트워크가 엄청 깊어져야 하고, 이를 메꾼게 dilation.

gated activation 을 강조한거 보니까 이게 꽤 유의미 했나봄.

### 2.1. Higher Fidelity WaveNet

audio quality를 높이기 위해 mu-law 16비트로 쓰고 softmax 대신에 discretized mixture of logistics distribution 을 활용함. 또한, 16kHz에서 24kHz로 sample rate도 높임. 보다 넓은 receptive field를 위해 filter size도 3으로 키움. 

## 3. Parallel WaveNet

IAF는 stochastic generative model 이고 latent variables에 의해 모든 sample이 parallel 하게 학습될 수 있음.

IAF는 normalising flow는 multivariate $P_X(x)$ 랑 explicit invertible non-linear f, tractable latent $P_Z(z)$ (주로 isotropic gaussian)을 가지고 log prob을 계산.

$logP_X(x)=logP_Z(z)-log|\frac{dx}{dz}|$

IAF 에서 $x_t$는 $p(x_t|z_{\le t})$ 이고, $x_t=f(z_{\le t})$로 계산됨. 그럼 jacobian은 triangular고, determinant는 diagonal term의 product.

$z\sim Logistic(0, I)$ 에서 샘플링. 이때 f를 invertible하게 만들기 위해 $x_t=z_t\cdot s(z_{<t},\theta)+\mu(z_{<t}, \theta)$ 로 두고 stddev s랑 mean mu 를 뉴럴넷으로 뽑는 방식.

## 4. Probability Density Distillation

## 5. Experiments

## 6. Conclusion

## 7. Acknowledgements

## 8. Citation

1. PixelCNN++ (discretized mixture of logistics distribution): Tim Salimans et al., 2017
2. Normalizing flows
    - Non-linear independent components estimation: Laurent Diinh et al., 2014
    - Density estimation using real nvp: Laurent Dinh et al., 2016.
    - Variational inference with normalizing flows: Danilo Jimenez etal., 2015.