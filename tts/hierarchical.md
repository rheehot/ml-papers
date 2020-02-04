# Hierarchical Generative Modeling for Controllable Speech Synthesis

Wei-Ning Hsu, Yu Zhang, Ron J. Weiss, Hiega Zen, Yonghui Wu, Yuxuan Wang, Yuan Cao, Ye Jia, Zhifeng Chen, Jonathan Shen, Patrick Nguyen, Ruoming Pang

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

이 페이퍼에서는 TTS 모델에서 annotate하기 쉽지 않은 latent attribute를 조정하는 방식을 제안함. 예로 speaking style, accent, background noise, recording conditions 등이 있음. 모델을 VAE 기반의 cond gen 모델로 가정할 건데, 이에 2 단계의 hierarchical latent를 이용할거임. 하나는 categorical로 속성 자체를 나타내게, 나머지는 그 속성의 정도를 나타내는 conditional한 distribution으로 표현할 것임. 그리고 이는 disentangled 되게 할 것. 이를 위해 GMM을 가져다 쓸 것. 그리고 앞서 이야기한 속성들이 실제로 조절 가능함을 보일 것. 

## 1. Introduction

근래에 연구된 TTS 모델들은 handcrafted feature 없이도 좋은 성능을 보였다. 그리고 이는 encoder, decoder 모델에 heavily하게 의존하고 있다. 그리고 이의 확장 버전으로 speaker identity 등의 attr을 cond로 하여 조정 가능한 모델들이 나왔다.

여기엔 speaker identity 외에 많은 속성들이 있고, 이는 라벨링 하기 어려움. 그리고 Skerry Ryan et al., 2018; Wang et al., 2018;은 이를 latent로 두고 auto encoder를 통해 모델링 하고자 함. 그리고 이를 적당히 잘 모델링 할 수 있게 됨. 

현재 crowdsource에 의해 여러 latent attr를 가진 데이터들이 많이 생겼고, 이를 학습해서 latent repr를 dientangled할 수 있으면 이는 indep하게 조절할 수 있음. 그리고 이는 data augmentation 등의 부분에서 다양히 이용될 수 있음.

Taco2를 확장해서 만들꺼고, labeld attributes와 unlabled attributes로 나눠서 학습. 각각의 latent는 vae framework랑 gmm prior를 활용. 결과 latent space는
1. disentangled attr repr을 배울 수 있었고
2. interpretable clusters가 존재했음
3. posterior로 부터 sampling이 가능함.

contribution은 다음과 같음.
1. probabilistic hierarchical generative model 을 제안할거고, sampling stability와 disentangled attribute control을 제공함. 또한 interpretability와 quality가 타 논문과 비교 가능할 정도였음.
2. model formulation은 latent 인코딩을 명시적으로 분리하는데, 이 때 supervised speaker attr과 latent attr을 2개의 mixture dist로 분리해서 학습함. 그리고 이는 직관적인 conditioning을 가능케 함.
3. 이는 첫번째로 high quality controllable tts system을 만든 논문임. 

## 2. Model

Tacotron-like TTS는 text seq $Y_t$ 와 observed cat label $y_o$를 input으로 쓰고, ar dec를 통해 acoustic feat X를 frame by frame으로 추측해 나감. mse를 rctor loss로 쓰고 이는 사실 $p(X|Y_t, y_o)=\prod_np(x_n|x_1,...,x_{n-1}, Y_t, y_o)$ 의 mle를 구하는 것과 같음. 이는 isotropic gaussian으로 가정하고 mean을 추측하는 것과 같음. 그리고 이는 unlabled latent를 high variance의 cond dist로 구성함. 그리고 이를 control하기 위해서는 graphical model과 hierarchical latent variables가 요구됨.

### 2.1. Conditional Generative Model with Hierarchical Latent Variables

$y_l$과 $z_l$은 추가적인 변수로 K-way cat discrete var인 $y_l$은 named attr class임. $z_l$은 D-dimensional cont variable, latent attr repr임. 이 페이퍼에서 $y_*$은 disc, $z_*$은 cont variables임. speech X를 ㅁ나들기 위해서 $Y_t$, $y_o$에 조건화 하고, 와 $y_l$을 prior에서 sampling, $z_l$을 conditional 에서 sampling함. 그리고 parameterized synthesizer nn에서 X를 sampling함.

$p(X, y_l, z_l | Y_t, y_o) = p(X|Y_t, y_o, z_l)p(z_l|y_l)p(y_l)$

이 때 $p(y_l)$을 non-informative prior, uniform $K^{-1}$을 가정함. 그리고 $p(z_l|y_l)=\mathcal N(\mu_{y_l}, \mathrm{diag}(\sigma_{y_l}))$ 로 가정. 그럼 이 때 marginal prior $z_l$은 diagonal cov와 동일한 mixture weight을 가지는 GMM임. 그럼 자연스래 cluster를 가지고 이에 따른 disentangled attr repr을 가지게 됨. 그리고 해당 클러스터에 모델이 적절히 training data의 attr을 할당할 것이라 가정. 그리고 diagonal cov는 uncorrelated factor을 가정하게 되므로 독립적인 sampling이 가능해짐.

### 2.2. Variational Inference and Training

$p(X|Y_t, y_o< z_l)$ 은 NN에 의해서 parameterized 됨. variational dist $q(y_l|X)q(z_l|X) 는 $p(y_l, z_l|X, Y_t, y_o)$를 approximate 하는데 이용됨. 이는 unseen attr이 텍스트와 observed attr과 독립임을 표현. $q(z_l|X)$는 diagonal cov gaussian으로 가정하고 mean과 var은 NN에 의해 모델링됨.

$p(y_l|X) = \int_{z_l}p(y_l|z_l)p(z_l|X)dz_l = \mathbb E_{p(z_l|X)}[p(y_l|z_l)] \simeq \mathbb E_{q(z_l|X)}[p(y_l|z_l)] := q(y_l|X)$

위에 따라 $p(y_l|X)$은 $q(y_l|X)$로 근사. ELBO로 학습

$\mathcal L(p, q;X,Y_t, y_o)=\mathbb E_{q(z_l|X)}[\log p(X|Y_t, y_o, z_l)] - \mathbb E_{q(y_l|Z)}[D_{KL}(q(z_l|X)\ ||\ p(z_l|y_l))] - D_{KL}(q(y_l|X)\ ||\ p(y_l))$

나머지는 reparametrization trick에 의해 differentiable하고, $q(z_l|X)$는 Monte carlo sampling으로 근사.

### 2.3. A Continuous Attribute Space for Categorical Observed Labels

observed label $y_o$가 주어졌을 때 observed attribute repr $z_o$에 대해 diagonal cov gaussian $p(z_o|y_o) = \mathcal N(\mu_{y_o},\mathrm{diag}(\sigma_{y_o}))$로 가정. 이에 ELBO는 다음과 같음.

$\mathcal L(p, q;X,Y_t, y_o)=\mathbb E_{q(z_o|X)q(z_l|X)}[\log p(X|Y_t, z_o, z_l)] - D_{KL}(q(z_o|X)\ ||\ p(z_o|y_o)) - \mathbb E_{q(y_l|Z)}[D_{KL}(q(z_l|X)\ ||\ p(z_l|y_l))] - D_{KL}(q(y_l|X)\ ||\ p(y_l))$

이 때 disentangled observed attr을 위해 $p(z_o|y_o)$ 의 분산을 작게 잡음.

### 2.4. Neural Network Architecture

$p(X|Y_t, z_o, z_l), q(z_o|X), q(z_l|X)$ 3개를 각각 synthesizer, observed encoder, latent encoder로 가정. synthesizer은 taco2에 따름. 이 때 latent는 decoder input에 concatenate 되어 들어감. 

## 3. Related works

## 4. Experiments

## 5. Conclusion

## 6. Citation
