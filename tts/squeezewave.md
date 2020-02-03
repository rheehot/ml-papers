# SqueezeWave: Extremely Lightweight Vocoder for On-Device Speech Synthesis

Bohan Zhai, Tianren Gao, Flora Xue, Daniel Rothchild, Bichen Wu, Joseph E. Gonzalez, Kurt Keutzer.

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

automatic speech synthesis는 단말기에서 사용자와 음성을 통해 소통하기 위한 중요한 하나의 분야가 되었음. 현재 존재하는 대부분의 보코더는 autoregressiveness에 의해 병렬화가 어려움. WaveGlow는 autoregressive하지 않은 flow based 모델임. WaveGlow는 parellelizable하지만 여전히 edge device에서 돌기엔 complex함. SqueezeWave는 waveglow based로 한 경량화 모델 

## 1. Introduction

edge는 네비게이션, home assistant, translation app등 speech 기반의 의사소통이 늘어났고, 이를 기반으로 한게 TTS 알고리즘. 이런 근래의 TTS 모델으 complex하고 computing resource를 굉장히 많이 요구함. 그렇기에 주로 cloud에서 연산해서 보내주는 방식.

몇몇 트렌드는 이걸 더 어렵게 한다.
1. 모바일 기종의 computability 향상으로 클라우드가 짊어져야 할 cost를 줄일 수 있게 함.
2. data privacy에 대한 인지 향상으로 사용자들이 개인 정보를 클라우드로 주고 받는걸 원치 않음.
3. low latency를 요구함. 

modern tts는 acoustic model과 vocoder로 나뉨. 여기서 이야기 하고 싶은건 vocoder efficiency를 늘이고 싶은 것. wavenet 같은건 너무 complex함. 또한 autoregressive nature에 의해 parellism을 어렵게 함. waveglow는 non-auto-regressive. highly parallelizable 하지만 여전히 computationally expensive. waveglow를 optimizing 하는 것을 목표로 함. 

## 2. Computational Complexity of WaveGlow

## 3. SqueezeWave

## 4. Experiments

## 5. Citation
