# ITFTE: Improved Time-Frequency Trajectory Excitation Vocoder for DNN-Based Speech Synthesis

Eunwoo Song, Frank K Soong, Hong-Goo Kang

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

이 논문에서는 DNN-based SPSS를 위한 ITFTE 보코더에 대해서 연구할 것. ITFTE는 LPC 기반의 보코더로 time-freq domain에서 periodicity dist로 pitch-dependent excitation signal이 표현된다. 제안하는 메소드는 ITFTE 보코더의 parameterization 효율성을 증대했다. DCT의 orthogonality를 통해 ITFTE의 복원률과 perceptual quality를 높였다.

## 1. Introduction

SPSS는 DNN 기반의 training process와 접합될 때 더욱 발전됐다. centralized network는 input context와 output acoustic features 사이의 복잡한 관계를 compact하게 모델링 할 수 있게 했다. 이는 정확도가 올랐을 뿐 아니라 생성된 파라미터의 smoothing problem도 해결했다. 이는 HMM 보다 훨씬 좋은 성능을 보였다.

이전 연구에서는 SPSS + TFTE 모델에 대해 이야기함. pitch-dependent ext signal을 slowly evolving waveform (SEW)와 rapidly (REW)로 분리함으로써 TFTE 모델은 periodicity distribution을 time-freq domain에서 ext signal로부터 분리해냄. SEW는 TFTE에서 제일 중요한데, 이는 ext에서 quasi-periodict/voiced portion을 나타냄. 반면 REW는 noise-like component를 나타냄. SEW와 REW를 통해 다향한 phonetic의 시간축상 periodicity를 조절할 수 있어 TFTE는 BAP 보다 좋은 성능을 보인다.

ITFTE는 TFTE를 위한 parameterization method를 제공함. SEW와 REW는 DNN에 바로 적용할 수 없는데 이는 parametric dim이 pitch interval의 길이에 의존적이기 때문이다. ITFTE 보코더에서 SEW는 고정된 수의 frequency sub-bands로 나뉘고, DCT로 변환된다. 각 sub band의 첫번째 DCT component는 펴균을 의미하고, 이를 DNN에 이용한다. 나머지 component는 gaussian rand var이라 봐도 무관하다. REW는 power contour estimation method로 모델링 되는데 이는 REW 자체에 의한 perceptual difference가 크지 않기 때문. 이를 통해 dimensional variation 문제를 해결한다.

전체적인 개선을 위해 full-band DCT-based parameterization 메소드를 제안함. 일전의 sub-band DCT로 접근은 모든 sub-bands가 full-band SEW로 combine될 때 투명치 못한 소리를 내는데, 이는 random var에서 sampling된 component는 충분한 정보나 복원률을 보이지 않기 때문이다. 제안하고자 하는 방법론에서는 SEW를 subbands로 나누기 보단 fullband에 DCT를 취한다. DCT의 energy compactness, invertibility property에 의해 쉽고 정확하게 고정된 수의 DCT coeff에서 복원할 수 있다. 이렇게 복원된 SEW는 DCT basis의 orthogonality에 의해 smoother해진다. ITFTE의 복원율응 굉장히 많이 줄었고, synthesis 퀄리티를 높였다.

## 2. Related works

## 3. Model description

## 4. Experiment

## 5. Future works

## 6. Citation
