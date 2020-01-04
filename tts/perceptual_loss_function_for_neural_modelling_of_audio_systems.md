# Perceptual loss function for neural modelling of audio systems

Alec Wright and Vesa Valimaki

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

이 논문에서는 non-linear nn based audio system에서 pre-emphasis filter를 이용하는 방식을 제안함. 이전 연구에서는 first-order highpass pre-emphasis filter를 통한 error2signal ratio loss func를 이용했다면, 이번 연구는 보다 perceptual 하게 연관이 있는 pre-emphasis filter를 찾아보는 것. 그리고 이게 lowpass filtering at high freq 관련인가봄. 결과 기타 사운드에서 A-weighting pre-emphasis filter가 가장 효율적이었음. 그리고 computational cost가 O(1) 수준임.

## 1. Introduction

음악 관련 도구 중에서 추가 음향 효과를 줬을 때 이걸 emulate 하는게 어려움. 그래서 쓰는게 rnn based nonlinear audio circuits 를 쓰는거.

## 2. Related works

## 3. Model description

## 4. Experiment

## 5. Future works

## 6. Citation
