# Reformer: The Efficient Transformer

Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya.

## Summary

- one-line summary
- topic words : 
- base model : 
- variation : 
- benefits :
- weakness :
- future works :

## Abstract

Large Transformer은 다양한 분야에서 SOTA를 이뤘지만 특히 긴 시퀸스에 대해 굉장히 무거운 모델임. 이 논문에서는 이를 효율적으로 다루기 위한 방법을 이야기하며, 예로 dot-product대신에 locality-sensitive hash를 쓸 것임. 또한 reversible residual를 이용해서 activation을 한번만 이용할 수 있게 함. 그리고 이를 reformer라 명명.

## 1. Introduction

transformer는 nlp 다양한 분야에서 sota를 만들어냈다. 그 중 제일 근건 레이어당 0.5B의 파라머와 64개의 레이어를 쌓기도 했다. 또한 1만여개의 symbol을 하나의 문장에 넣어 학습하기도 했었다. 음악이나 이미지 분야에서도 시도가 있었으며, large-scale long-seq 모델은 좋은 성능을 보이면서도 nlp 자체의 연구를 저해한다는 지적도 있었다. 이러한 모델은 single gpu로는 fine tuning도 안된다.

트랜스포머에서 가장 메모리를 많이 쓰는 부분은 다음들이다.
- activation와 back prop용 텐서
- feed-forward layer에서의 depth 문제
- dot prod attention이 L제곱 먹는거

이를 reformer에서 해결하기 위해 다음을 도입.
- reversible layer를 통해 activation의 single copy만을 저장
- FF layer 내부의 activation을 split
- locality sensitive hashing으로 LlogL 시간에 attention

그리고 이것들이 실제로 training process에 미미한 영향을 끼침을 보일 것이다. spliting activation은 nuermical하게 동일한 연산이고, reversible residual 또한 큰 영향을 미치지 않았다. 또한 locality-sensitive hashing이 training dynamics에 영향을 줄 것인데, 얼마나 많은 헤드가 필요한지 등의 값을 찾아 볼 것.

여러 실험을 통해 transformer과 비슷하지만 더 빠른 성능을 보임을 보일 것이다.

## 2. Locality-Sensitive Hashing Attention

**Dot-proeudct attention** $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

**Multi-head attention**  transformer는 single attention 대신에 key, query, value를 여러개로 projection하여 각각을 attention 취하는 방식. 그리고 이를 다시 concat해서 projection.

**Memory-efficient attention** 위의 가장 큰 문제는 $QK^T$의 매트릭스가 LxL 인데, L=64K에 32bit float이면 이만 16기가를 먹음. 사실 이는 모두 메모리에 올라갈 필요가 없는데 $\mathrm{softmax}(\frac{q_iK^T}{\sqrt{d_k}})V$ 를 쓰면 한번에 L 만큼의 길이만 저장하고 있을 수 있음. 그리고 이는 필요한 경우 backprop 때 다신 연산함으로써 메모리를 더 단축 시킬 수 있음.

**Where do Q, K, V come from?** Q, K, V는 실상 embedding vector등 A에서 projection 되어서 이용되는데, Q와 K가 같다는 constraint를 건다면 projection layer를 두개만 쓰면 됨. 그리고 이를 앞으로는 shared-QK transformer이라고 할 것.

**Hashing attention** high dim에서 KNN을 찾는 빠른 방법 중에는 locality-sensitive hashing이 있다. LSH의 경우 가까운 벡터에 대해 동일한 hash를 가질 확률이 높고, 멀수록 확률이 떨어진다. 여기서는 LSH가 가까운 벡터에 대해 높은 확률로 동일한 해시를 주고, 버킷의 크기가 거의 비슷해지도록 constraint를 건다.

이를 논문에서는 random matrix R를 이용해 $h(x) = \mathrm{argmax}([xR; -xR])$ 로 projection, concat 하여 이용한다. 이는 LSH scheme (Andoni et al. 2015) 에서 발췌.

**LSH attention** 먼저 normal attention을 i-th single query에 대해서 rewriting.

$o_i=\sum_{j\in \mathcal P_i}\exp(q_i \dot k_j - z(i, \mathcal P_i))v_j \ \mathrm{where} \ \mathcal P_i = \{j:i\ge j\}$

여기서 $\mathcal P_i$는 i번째 query가 attend 하는 집합, z는 partition function, 에로 normalizing term. scaling 텀은 clarity를 위해 제외.

batch 기능을 위해 $\mathcal P_i$ 보다 큰 set으로 masking

$o_i = \sum_{j\in \hat \mathcal P_i}\exp(q_i\dot k_j - m(j, \mathcal P_i) - z(i, \mathcal P_i))v_j \ \mathrm{where}\ m(j, \mathcal P_i) = \infty \ \mathrm{if} \ j \notin \mathcal P_i \ \mathrm{otherwise} \ 0$

이 때 $\mathcal P_i$를 restricting 해서 $q_i$ 가 attend 할 수 있는 대상을 제한하면, hash bucket에 대해서만 attention 연산을 할 수 있게 됨.

$\mathcal P_i = \{ j:h(q_i)=h(k_j) \}$

full attention을 취할 경우 matrix가 sparse해도 sparsity를 충분히 활용하지 못하고 전체에 대해 attention을 취해야 함. 이를 hash bucket에 따라 sorting하면 similar item이 모이게 되고, full attention pattern이 bucket에 대한 attention으로 근사됨. 

이 때 hash bucket의 크기는 고르지 않을 수 있고, 이는 batch를 하기 어렵게 함. 게다가 query len이랑 key len이 다르면 버킷이 query는 많은데 key가 없는 현상이 생길 수 있음. 이를 해결하기 위해 먼저 $h(k_j) = h(q_j)$ 이기 위해 $k_j=\frac{q_j}{||q_j||}$ 로 설정해야 하고, 정렬을 할 때는 bucket 번호대로, sequence 상대위치대로 진행해야 한다. 이렇게 정렬되고 나면 버킷들은 diagonal term 근방으로 모이게 됨. 그리고 나서 m개 쿼리씩 잘라서 chunk를 만들면 batch도 가능해짐. 각 chunk는 자신과 attention하고 하나의 다음 청크와 attention 함. 

$\hat \mathcal P_i = \left\{ j:\lfloor \frac{s_i}{m} \rfloor - 1 \le \lfloor \frac{s_j}{m}\rfloor \le \lfloor \frac{s_i}{m} \rfloor\right\}$

만약 $\max_i|\mathcal P_i| < m$ 이면 $\mathcal P_i \subseteq \hat \mathcal P_i$ 가 만족. 실험적으로 $m=\frac{2l}{n_{buckets}}$ 으로 설정. 평균적으로 버킷 크기가 $\frac{1}{n_{buckets}}$이고, 확률적으로 두배 이상 커지진 않을거라 가정하였으므로.

**Multi-round LSH attention** 여전히 다른 버킷에 유사한 아이템이 떨어질 수 있기 때문에 여러 단계의 hashing을 진행.

$\mathcal P_i = \bigcup^{n_{rounds}}_{r=1}\mathcal P^{(r)}_i \\ \mathrm{where}\ \mathcal P^{(r)}_i = \left\{ j : h^{(r)}(q_i) = h^{(r)}(q_j) \right\}$

이는 병렬적으로 실행

**Causal masking for shared-QK attention** transformer decoder에서 미래 정보를 차단하기 위해 query key를 lsh attention 연산후 reordering 해서 mask를 연산함.

몇몇 impl에서는 자기 자신에 대한 attention까지는 허용하기도 함. 이건 shared QK formulation에서 썩 좋지 않은데, 자기자신과의 attention 값이 다른 query 와의 attention 보다 거의 대부분 크기 때문. 그래서 현재에는 자신을 attending하는건 막고 있음.

| Attention Type | Memory Complexity | Time Complexity |
|---|---|---|
| Scaled Dot-Product | $\max(bn_hld_k, bn_hl^2)$ | $\max(bn_hld_k, bn_hl^2)$ |
| Memory-Efficient | $\max(bn_hld_k, bn_hl^2)$ | $\max(bn_hld_k, bn_hl^2)$ |
| LSH Attention | $\max(bn_hld_k, bn_hln_r(4l/n_c)^2)$ | $\max(bn_hld_k, bn_hn_rl(4l/n_c)^2)$ |

### 2.1. Analysis on a synthetic task

해쉬 수가 적을 때보단 많을 때 성능이 좋았음.

## 3. Reversible Transformer

어텐션 연산량이 많이 줄었지만 여전히 bnl로 시작하는 coeff를 무시할 수 없음. 여기에 activation과 feed-forward layer까지 들어가면 여기서만 또 몇기가씩 메모리를 먹음. 이를 해결하기 위해 도입한게 reversible layer.

**RevNets**. Reversible residual net은 Gomez etal. 2017. 에서 소개됨. main idea는 any layer의 액티베이션이 다음 레이어들에서 복원될 수 있는 것이다. backprop을 위한 checkpoint를 두기 보다는 one-by-one으로 reverse하면서 back prop이 가능해진다. 

$y_1 = x_1 + F(x_2) \\ y_2 = x_2 + G(y_1)$

에 대해 다음과 같이 reverse

$x_2 = y_2 - G(y_1) \\ x_1 = y_1 - F(x_2)$

**Reversible Transformer** RevNet block에 F대신에 attention, G 대신에 feed-forward로 대치. 

**Chunking** FeedForward 레이어가 position independent한 operation을 취하기 때문에 chunk로 쪼개서 연산할 수 있음.

$Y_2 = \left[ Y^{(1)}_2;...;Y^{(c)}_2\right] = \left[ X^{(1)}_2 + FeedForward(Y^{(1)}_1); ...;X^{(c)}_2 + FeedForward(Y^{(c)}_1) \right]$

이후는 batching 해서 parallel하게 돌려버리면 됨. 아님 한번에 chunk 한번씩 돌려서 메모리를 줄일 수도. backward도 parallel하게 돌 수 있음. 

**Chunking, large batches and parameter reuse** 필요에 따라 안 쓰는 파라미터를 CPU에 올리기도 함. 

## 4. Related works

## 5. Experiments

## 6. Reference
- LSH scheme: Andoni et al., 2015.
- Reversible residual networks: Gomez et al., 2017.