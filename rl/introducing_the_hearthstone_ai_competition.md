# Introducing the Hearthstone-AI Competition

Alexander Dockhorn, Sanaz Mostaghim.

## Summary

- 하스스톤 대회 소개
- topic words : POState, complexity, randomness, meta-game
- base model : -
- variation : -
- benefits : -
- weakness : -
- future works : 다른 여러 대회를 통해 AI 발전을 도모하자

## Abstract

Hearthstone AI framework와 관련 대회들은 AI가 collectible card games를 도전하게 만듬. 이런 problem의 특징은 카드의 수가 많고, 사용자가 골라 덱을 구성할 수 있음. 이는 synergies 라는 또 다른 개념을 도입하게 함. 가능한 덱의 수도 엄청나고, randomness, restricted information은 에이전트 개발을 어렵게 함. 이번 페이퍼에서는 problem, challenges 등을 소개하고자 함.

## 1. Introduction

알파고 등의 game ai 성공에 의해 미디어의 주목을 다시 받고 있는게 요즘. 게임은 challenging 하고 well-balanced problems를 제공하며, 이는 새로운 에이전트 아키텍쳐 개발을 수요로 함. 그리고 competitive nature에 의해 여러개의 agent와 여러 다른 조건에 대한 비교 대상으로써의 ideal한 test-bed를 제공함. 

게임 관련 연구는 그에 맞는 에이전트를 만들게 했는데, 대표적으로 체스, 바둑, 포커, 팩맨, 스타 등이 있음. 이렇게 개개로 만드는 경우도 있고, 일반화하고자 하는 경우도 있음. Arcade Learning Environment framework, general video game ai framework. 

이번 페이퍼에서는 하스스톤 ai competition에 대해 소개할 것. competition의 목적은 하스스톤을 자동으로 플레이 하는 에이전트를 만드는 것. collectible card games의 대표적인 특징들은 다음과 같음.

- **Partial observable state space**: 게임의 전체 state를 관찰할 수 없기에 플레이에 리스크를 감수해야 함. 
- **High complexity**: 대략 2000개가 넘는 카드가 서로 다른 고유의 효과를 가지기에 tree 구성에 복잡도가 증가함
- **Randomness**: 카드의 액션이 임의성을 동반함. 그렇기에 descrete한 planning이 어려워지고, 관측 결과에 따라 adapt할 수 있어야 함.
- **Deck-building**: 카드 간의 시너지가 있고 이를 탐색하여 카드의 효과가 적절히 날 수 있도록 해야함. 
- **Dynamic Meta-Game**: 사용자의 스킬이나 현재 덱에 fully-dep 하지 않고 어떤 덱이 자주 쓰이는지 어떤 플레이가 나오는지 등의 meta를 학습할 여지가 있음.

## 2. Hearthstone: Heros of Warcraft

하스스톤 설명

## 3. Hearthstone-AI Competition Framework

Hearthstone-AI competition은 sabberstone을 기반으로 함. C#으로 만들어졌고, 에이전트가 스테이트에 접근하고 process하는데 도움을 주는 Helper class를 제공함. 

AbstractAgent 클래스를 상속 받아야 하고, InitializeAgent, FinalizeAgent 등의 함수들을 구현해야 함. GetMove를 통해 action을 취함. POGame (partially observed game) 객체를 상태로 받음. 60초의 계산 시간이 주어지고, 액션은 되돌릴 수 없음. 50턴 제한이 있고 그 안에 승부를 봐야 함.

POGameHandler은 시뮬레이션을 진행하고 report를 제공. 

## 4. Competition Tracks

- **Premade Deck playing-track**: 미리 만들어둔 6개 덱 중 하나를 골라 쓰는 것. 장기 목표는 어떤 덱이든 돌릴 수 있는 에이전트 만들기
- **User Created Deck Playing-track**: 사용자가 만든 덱을 쓰는 것. 덱에 맞는 strategy를 최적화하는 것이 목표

## 5. Conclusions and Future Plans

다음 트랙들도 만들어 보려 함.
- **Deck-building**: deck building process를 연구.
- **Draft mode deck-building**: 드래프트 모드에 쓰일 덱 만드는거
- **Game balancing/Card generation**: 카드 만들어 보는거.

## 6. Citation

- M.G. Bellemare et al., The Arcade Learning Environment: An Evaluation Platform for General Agents
- D. Perez-Liebana et al., The 2014 General Video Game Playing Competition
