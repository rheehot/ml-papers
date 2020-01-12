# Content Research

hspp 관련 논문 정리

## 0. HSPP 구상안

### Abstract

HS가 지금 위치하고 있는 RL 분야에서의 challenge를 소개. 이 환경으로 hspp를 제안했고, 이에 따른 HSPP의 목표, 지원하는 기능, 장점 요약.

### 1. Introduction

### 2. Architecture

### 3. Example

### 4. Experiments

### 5. Conclusion and Future work

### References

## 1. ELF An Extensive, Lightweight and Flexible Research Platform for Real-time Strategy Games

Yuandong Tian, Qucheng Gong, Wenling Shang, Yuxin Wu, C. Lawrence Zitnick

### Abstract

ELF: Extensive, Lighweight Flexible platform. 

highly customizable real-time strategy (RTS) engine 이라는거 보니까 프레임워크를 경량화 하고 확장성을 목표로 했나봄. Mini-RTS, CTF, Tower Defense 3개 지원. 여기에 modern rl  메소드를 coupling 시켰다는거 보니까 메소드 확장성은 부족할 듯. env-agent communication 방식. 기존의 C++ 기반 환경과도 소통 가능. 

### 1. Introduction

game env는 리얼 월드랑 다르게 controllable, reproducible, automatically labeld 한 ideal한 실험 환경으로 작용 가능.

1. Extensive: env는 리얼월드의 다양한 상황을 캡쳐할 수 있어야 하며, extensive feature set에 의해 env의 리얼월드로의 일반화가 가능해져야 함.

2. Lightweight: real-world에 비해 적은 리소스로 빠르게 샘플을 생성할 수 있어야 함.

3. Flexible: 다양한 환경 선택과, game parameter manipulation, internal variable accessibilities, algorithms, 다양한 레벨에서의 customizing을 지원

기존 아키텍쳐는 모든걸 다 지원하지 않음.

그래서 ELF 를 제안하고자 함. research-oriented 하고, 다양한 properties와, efficient simulation, highly customizable함. RL Method가 coupled되어 있어서 유연히 학습 가능. 

real-world scenario나 complex games은 자연적으로 hierarchy를 가짐. policy 학습도 top-level strategy과 low-level commands를 나눠 하는 방식으로 ?

ELF 는 Python/C++ 하이브리드를 지원해서 fully-python interface 보다 토폴로지를 바꾸거나 병렬성을 지원하는데 편리함. 

backend는 pytorch. A3C를 베이스로 하나봄. 

### 2. Architecture

ELF는 producer-consumer paradigm을 따름. C++ 모델로 여러개의 게임을 동시에 돌리고, python 모델 기반 consumer 에서 action을 학습, reply를 주는 방식. 그 아래는 작동 방식.

**Parallelism using C++ threads** 얘는 single game instance를 python wrap 한 방식이 아닌, multi game instance를 C++ 레벨에서 병렬화해서 배치 데이터를 파이썬으로 주는 방식. 

**Flexible Environment-Model configuration** 여러 속성들, 토폴로지 (one2one, multi2one) 등 병경 가능. 이를 통해 여러 deep rl method 연결이 가능해짐.

**Highly customizable and unified interface** 기존의 C++ 게임을 연결시킬 수도 있고, 사전에 만들어진 게임을 가져다 쓸 수도 있음. 

**ReinforcementLearning backend** pytorch 기반으로 A3C, Policy-Gradient, Q-learning 등이 사전에 만들어져 coupling 되어 있음. 

### 3. Real-time strategy Games

### 4. Experiments

### 5. Conclusion and Future works

ELF는 research-oriented platform for concurrent game simulation이고, flexible 한 controllability를 제공. 이걸 기반으로 RTS 환경을 만들어 봤고, 학습도 해봤음. 벤치 마크 결과 흥미로운 행동을 보임. 

이걸 토대로 더 많은 RTS 관련 RL 연구가 가능할 것.

## 2. StartCraft II: A New Challenge for Reinforcement Learning

Oriol Vinyals et al., 2017.

### Abstract

여기너 SC2LE (StarCraft II Learning Environment)를 소개. 이 분야는 rl의 새로운 도전과제임. multi-player, multi-agent problem, imperfect information, large action space, large state space, delayed credit 등. 스타2의 domain knowledge를 소개하고, python-based interface를 제공함. main game 뿐만 아니라, 스타2 의 또 다른 요소에 집중 할 수 있는 mini-game도 제공. 그리고 이에 따른 rl baseline을 제공. 물론 좋은 성능을 내지는 못함. sc2le가 이제 이 분야의 새로운 발판으로써 작용할 것.

### 1. Introduction



### 2. Related Work

### 3. The SC2LE environment

### 4. Reinforcement Learning: Baseline Agents

### 5. Supervised Learning from Replays

### 6. Conclusions & Future Work

## 3. The playStation Reinforcement Learning Environment

## 4. OpenSpiel: A Framework for Reinforcement Learning in Games

## 5. The Many AI Challenges of Hearthstone
