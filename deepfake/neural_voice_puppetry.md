# Neural Voice Puppetry: Audio-driven Facial Reenactment

Justus Thies, Mohamed Elgharib, Ayush Tewari, Christian Theobalt, Matthias Niebner.

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

Neural Voice Puppetry를 소개할거고, 이건 audio-driven facial video synthesis를 하기 위한 모델임. 오디오 시퀸스가 주어지면 이미지와 유사한 비디오를 만들어내는 모델. 딥러닝으로 할거고, 3D face latent space를 차용함. neural rendering으로 프레임을 뽑는 동안에 stability를 보장하는 역할을 함. 이러한 접근은 서로 다른 사람들 사이를 일반화 하고, unknown source 오디오와 비디오를 합칠 수 있게 해줌. TTS 오디오도 사용이 가능하게 된 것. 

## 1. Introduction

스피치 기반 소프트웨어가 자주 쓰이게 되었고, 여기엔 많은 머신러닝 접근을 기반으로 함. 이러한 virtual agents는 사용자 친화적인 man-machine interface를 제공하는 것. 하지만 비주얼적인 부분이 여전히 challenging하게 남음.

이런 challenge를 위해서 한게 Neural Voice Puppetry이고, 오디오에는 wavenet이랑 multi-speaker transfer learning을 이용했고, visual basis로는 실제 인물이 들어간 비디오를 사용함. 키포인트는 lipsync인데, audio에서 lip point를 예측하고, target pterson의 외모에 맞게 렌더링하는 방식. 많은 데이터를 요구하지 않고, 2~3분 정도의 짧은 타겟 비디오들로 학습 가능. 

다양한 사람들의 얼굴을 3D latent로 일반화하기 위해 노력했고, 이는 여러 오디오에 대해 일반화 한 것과 같은 이야기. 이는 facial reenactment의 일종이기도 한데, 이는 타겟 비디오를 소스 액터에 대해서 re-animating하는 기술. 다양한 접근이 있지만 리얼리티가 떨어지는 문제가 여전함. synthesizing obama 같은 논문이 있지만, 데이터를 너무 많이 요구함. 또한, generalized 되지 않음. 

정리
1. 짧은 비디오 (2min, 3min per target video)
2. person specific talking style 반영
3. tts로 텍스트부터 비디오까지 end2end 가능
4. neural redering으로 잘 나올 것.

## 2. Related works

## 3. Overview

모델은 크게 두개로 구성. latent expression vector을 예측하는 generalized network (audio-expression space). reenactment를 위해 모든 사람이 동일한 audio-expression 공간을 공유. audio-expression은 blendshape coeff of 3D model로 해석. 사람마다 다를 것이고, second-part에서 학습됨. 사람들의 얼굴 모션, 외모 등의 특이점을 표현함. facial motions는 delta-blendshape 이고, 이는 generic face template의 subspace 이도록 제약함. 첫번째 스테이지에서는 generalized and specialized components 학습에 집중함. 

## 4. Data

RGB scale 비디오랑 그에 싱크가 맞는 오디오가 필요. 512x512, 25fps.

**Training Corpus for the Audio2ExpressionNet** 데이터셋은 116개 비디오, 평균 1.7분, 302750 frames, 사람이 평온한 상태로 이야기하는 비디오를 가정.

**Target Sequences** 타겟은 2~3분 정도로 그냥 인터넷에서 긁어온 비디오.

### 4.1. Preprocessing

**3D Face Tracking:** statistical face model과 delta-blendshapes를 활용하여 3D latent space를 구성. shape 100개, albedo param 100개, expressions 76개로 100몇개의 파라미터로 구성. Thies의 dense face tracking을 통해 model parameters를 모든 프레임에 걸쳐 구함. 렌더링을 위해 rasterized texture coordinates of the reconstructed face mesh도 저장.

**Audio-feature Extraction:** pretrained speech-to-text 모델 DeepSpeech 를 가져다가 피쳐를 뽑음. 또한 Voca와 같이 video frame에서 window of character logits를 뽑음. 그럼 16x29개 피쳐가 나옴. DeepSpeech는 모질라의 common voice dataset에서 학습해서 다양한 목소리에 대해 일반화됨. 

## 5. Method

3D face model을 face motion의 IR로 활용. 키포인트는 audio-based expression estimation. 사람마다 표정 표현이 다르기 때문에 모든 시퀸스에 대해서 person-specific expression spaces를 구성함. 그리고 latent audio-expression space는 공유. 그럼 audio-expression space에서 person-specific expression space로 전달됨. 여기에 기존 피쳐를 붙여 deferred neural rendering techniques로 이미지를 만들어냄. 

### 5.1. Audio2ExpressionNet

temporally smooth한 face motiond을 만드는게 목표. 이에 2단계 NN을 썼는데, 먼저 frame레벨에서 facial expression prediction을 진행. 이건 상당히 Noisy할 수 있는데, 이를 temporal filtering network를 통해 smooth expression을 만들어냄. 이는 동시에 학습이 가능하고, audio-expression coefficient를 뱉어냄. 

**Per-frame Audio-Expresion Estimation Network**

DeepSpeech의 RNN-part를 통해 feature를 뽑았고, 이는 20ms당 알파벳의 logits를 반환. 이 때 알파벳 29개에 time window를 16으로 잡아서 16 x 29 크기의 텐서가 나옴. 이 피쳐를 unfiltered audio-exp space로 매피하기 위해 4x conv, 3x fc 네트워크를 패싱하는데 2D conv에 stride (2, 1)를 써서 time dimension을 줄이는 역할을 함. 마지막엔 32개 tanh 값이 latent로 떨어짐

**Temporally Stable Audio-Expression Estimation**

smoothing을 하기 위해 filtering network를 쓰는데, T개 frame features를 시점 t에 대해 t-T/2 ~ t+T/2 로 가져옴. 이는 linear combination을 통해 하나의 결과를 뽑을건데, 이 때 쓸 weight을 뽑는게 filtering network가 할 일. dynamic fully-connected의 느낌. 5개 conv1d에 fc + softmax로. 이는 self-attention에서 받은 느낌임.

**Person-specific Expressions**

audio-expression spae에서 3D modeling을 하기 위해서 person-specific audio expression blendshape basis를 학습해야 함. 

audio-expression blendshapes는 generic blendshapes의 linear combination과 동치인데, 이로 인해 generalized network에서 generic blendshape basis로 가는 linear mapping이 되고, 이는 person specific하게 떨어짐. 결론은 이 matrix가 중요해지는거.

**Loss:** visually tracked training corpus를 통해 학습. verte-based loss function을 가정하였고, mouth region에 10배 정도의 가중치를 둠. vertex-wise distance를 구했고, 이를 rmse로 로스화

$L_{expr} = RMS(v_t - v_t^{*}) + \lambda L_{temp}$

$v_t$는 filtered expression estimated., $v_t^{*}$는 visual tracked vertices. temporal smoothness를 위해 이전 타입 스텝하고의 energy 차이도 이용함.

$L_{temp} = RMS((v_t - v_{t-1}) - (v_t^{*} - v_{t-1}^*)) + RMS((v_{t+1} - v_{t}) - (v_{t+1}^{*} - v_{t}^*)) + RMS((v_{t+1} - v_{t-1}) - (v_{t+1}^{*} - v_{t-1}^*))$

lambda는 20으로 실험함.

### 5.2. Neural Face Rendering

Image synthesis using neural textures 에서 나왔듯 얼굴 외모에서 뽑은 neural texture 기반 렌더링을 진행함. audio-driven expression estimations 기반으로 얼굴 하단을 synthesizing.

두개 네트워크가 쓰이는데 하나는 face interior용, 하나는 redering된 모델을 이미지로 뽑아내는거. target 이미지의 rigit pose를 가지고 렌더링. texture resolution은 256x256x16, 이걸 RGB 컬러로 뽑아내는거. U-Net 기반으로 돌지만 stride 대신 dilation 활용. visual artifacts를 날릴 수 있지만 파라미터 수가 늘어나진 않음. 배경에 잘 묻히기 위해서 얼굴 부분은 날리기도.

**Loss** frame per loss 를 쓰고, l1 두개를 하나는 최종 결과랑 GT, 하나는 face interior이랑 GT로 놓고, VGG 스타일 로스도 취함.

## 6. Results

### 6.5. Ablation studies

## 7. Limitations

## 8. Conclusion

## 9. Acknowledgements

## Appendix

## Citation
