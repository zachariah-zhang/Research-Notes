#### Introduction

Domain adaptation (DA), which aims at transferring knowledge from a labeled source domain to an unlabeled target domain, is a more economical and practical option than annotating sufficient target samples, especially in the keypoint detection tasks.

To solve the issues caused by the large ouput space, we delved into the predictions of a source-only keypoint detection model. We observed that when the predictions on the unlabeled domain are wrong, they are *not equally distributed* on the image.

This unexpected observation reveals that the output space is sparse in the sense of probability. Consider an extremely 6780 sparse case where the predicted position is always located at a keypoint, then a specific ankle detection problem becomes a K-way classification problem, and we can reduce the domain gap by enlarging the decision boundary between keypoints. This extreme case gives us a strong hint that if we can constrain the output space from a whole image space into a smaller one with only K keypoints, it may be possible to bridge the gap between regression and classification.

We first use an adversarial regressor to maximize the disparity on the target domain and train a feature generator to minimize this disparity. Based on the aforementioned observations and analyses, we introduce a spatial probability distribution to describe the sparsity and use it to guide the optimization of the adversarial regressor. It can somewhat avoid the problems caused by the large output space and reduce the gap between keypoint detection and classification in domain adaptation.

To this end, we convert the minimax game in DD into minimization of two opposite goals. This conversion has effectively overcome the optimization difficulty of adversarial training in RegDA.

#### Preliminaries

##### Learning Setup

In supervised 2D keypoint detection, we have n labeled samples $\{(x^i , y^i )\} ^n_ {i=1}$ from $X × Y^K$ , where $X ∈ R^{H×W×3}$ is the input space, which $Y ∈ R^2$ is the output space and $K$ is the number of keypoints for each input. The samples independently drawn from the distribution D are denoted as $\hat{D}$. The goal is to find a regressor $f ∈ F$ that has the lowest error rate $err_D = \mathbb{E}_{(x,y)∼D}L(f(x), y)$ on $D$, where $L$ is a loss function. 

In unsupervised domain adaptation, there exists a labeled source domain $\hat{P} = \{(x _s^ i , y _s^ i )\}^ n _{i=1} $and an unlabeled target domain $\hat{Q} = \{x^ t_ i \}^ m_{ i=1}$. The objective is to minimize $err_Q$.

##### Disparity Discrepancy

**Definition 1** (Disparity). Given two hypothesis $f, f' ∈ \mathcal{F}$ , we define the disparity between them as 
$$
disp_D(f',f)\triangleq \mathbb{E}_DL(f',f)
$$

**Definition 2** (Disparity Discrepancy). Given a hypothesis space $\mathcal{F}$ and a specific regressor $f ∈\mathcal{ F}$, the Disparity Discrepancy (DD) is defined by
$$
d_{f,\mathcal{F}}(P,Q)\triangleq \mathop{sup}\limits_{f'\in \mathcal{F}}(disp_Q(f',f)-disp_P(f',f))
$$
It has been proved that when $L$ satisfies the triangle inequality, the expected error $err_Q(f)$ on the target domain is strictly bounded by the sum of four terms: empirical error on the source domain $err_{\hat{P}}(f)$, empirical disparity discrepancy $d_{f,\mathcal{F}} (\hat{P} , \hat{ Q})$ between source and target, the ideal error $λ$ and complexity terms. Thus our task becomes 
$$
\min\limits_{f\in\mathcal{F}}err_{\hat{p}}(f)+d_{f,\mathcal{F}}(\hat{P},\hat{Q})
$$
We train a feature generator network $ψ$ which takes inputs $x$, and regressor networks $f$ and $f ′$ which take features from $ψ$. We approximate the supremum in Equation (2) by maximizing disparity discrepancy (DD):
$$
\max\limits_{f'}
D(\hat{P} , \hat{Q}) = \mathbb{E}_{x^t∼\hat{Q}}L((f'
◦ ψ)(x^
t
),(f ◦ ψ)(x^
t
))
− \mathbb{E}_{x^s∼\hat{P}}L((f'
◦ ψ)(x^
s
),(f ◦ ψ)(x^
s
))
$$


![Screenshot 2022-10-16 at 02.39.36](/Users/xinweizhang/Library/Application Support/typora-user-images/Screenshot 2022-10-16 at 02.39.36.png)

When the regressor $f ′$ is close to the supremum, minimizing the following terms will decrease errQ effectively,
$$
\min_{
ψ,f}
\mathbb{E}_{(xs,ys)∼\hat{P}}L((f ◦ ψ)(x
^s
), y^
s
) + ηD(\hat{P} , \hat{ Q})
$$

#### Method

##### Supervised Keypoint Detection

Most top-performing methods on keypoint detection generate a likelihood heatmap $\mathcal{H}(y_k ) ∈ R^{H'×W'}$ for each keypoint $y_k$ . The heatmap usually has a 2D Gaussian blob centered on the ground truth location $y_k$ . Then we can use L2 distance to measure the difference between the predicted heatmap $f(x^ s )$ and the ground truth $\mathcal{H}(y^ s )$. The final prediction is the point with the maximum probability in the predicted map $h_k$, i.e.$ \mathcal{J} (h_k) = arg max_{y∈\mathcal{Y}} h_k(y)$. 

Heatmap learning shows good performance in the supervised setting. However, when we apply it to the minimax game for domain adaptation, we empirically find that it will lead to a numerical explosion. The reason is that $f(x^ t )$ is not bounded, and the maximization will increase the value at all positions on the predicted heatmap.

To overcome this issue, we first define the spatial probability distribution $\mathcal{P}_T(y_k )$, which normalizes the heatmap $\mathcal{H}(y^k )$ over the spatial dimension,
$$
\mathcal{P}_T(y_k
)_{h,w} =\frac{
\mathcal{H}(y_k
)_{h,w}}
{\sum^{H'}_{h'=1}\sum^{W'}_{w'=1}\mathcal{H}(y_k)_{h',w'}}
$$
Donate by $\sigma$ the spatial softmax function,
$$
\sigma(z)_{h,w}=\frac{exp(z_{h,w})}{\sum^{H'}_{h'=1}\sum^{W'}_{w'=1}exp(z_{h',w'})}
$$
Then we can use the KL-divergence to measure the difference between the predicted spatial probability $\hat{p} ^s = (σ ◦ f)(x ^s ) ∈ R^{K×H×W}$ and the ground truth label $y^ s$,
$$
L_T(p^s,y^s)\triangleq\frac{1}{K}\sum\limits_k^KKL(\mathcal{P}_T(y^s_k)||p^s_k)
$$


In supervised setting, models trained with KL-divergence achieve comparable performance with models trained with L2 loss since both models are provided with pixel-level supervision. Since $σ(z)$ sums to 1 in the spatial dimension, the maximization of $L_T(p^ s , y^ s )$ will not cause the numerical explosion. In our next discussion, KL is used by default.

#####  Sparsity of the Spatial Density

*When the input is given, the output space, in the sense of probability, is not uniform.* 

This spatial density is sparse, i.e., some positions have a larger probability while most positions have a probability close to zero. To explore this space more efficiently, $f '$ should pay more attention to positions with high probability. Since wrong predictions are often located at other keypoints, we sum up their heatmaps,
$$
\mathcal{H}_F(\hat{y_k})_{h,w}=\sum\limits_{k'\not = k}\mathcal{H}(\hat{y_k})_{h,w}
$$
where $\mathcal{y}_k$ is the prediction by the main regressor $f$. Then we normalize the map $\mathcal{H}_F(y_k )$ into ground false distribution,
$$
\mathcal{P}_F(\hat{y_k})_{h,w}=\frac{\mathcal{H}_F(\hat{y_k})_{h,w}}{\sum_{h'=1}^{H'}\sum_{w'=1}^{W'}\mathcal{H}_F(\hat{y_k})_{h,w}}
$$
We use $P_F(\hat{y}_k )$ to approximate the spatial probability distribution that the model makes mistakes at different locations and we will use it to guide the exploration of $f '$. The size of the output space of the adversarial regressor is reduced in the sense of expectation. Essentially, we are making use of the sparsity of the spatial density to ease the minimax game in a high-dimensional output space.

##### Minimax of Target Disparity

Besides the problem discussed above, there is still one problem in the minimax game of the target disparity. Theoretically, the minimization of KL-divergence between two distributions is unambiguous. As the probability of each location in the space gets closer, two probability distributions will also get closer. But the maximization of KL-divergence will lead to uncertain results. Because there are many situations where the two distributions are different, for instance, the variance is different or the mean is different.

We hope that after maximizing the target disparity, there is a big difference between the mean of the predicted distribution. However, experiments show that $\hat{y}'$ and $\hat{y}$ are almost the same during the adversarial training. In other words, maximizing KL mainly changes the variance of the output distribution.

The reason is that KL is calculated point by point in the space. When we maximize KL, the probability value of the peak point is reduced, and the probability of other positions will increase uniformly. Ultimately the variance of the output distribution increases, but the mean of the distribution does not change significantly, which is completely inconsistent with our expected behavior. Since the final predictions of $f'$ and $f$ are almost the same, it’s hard for $f'$ to detect target samples that deviate from the support of the source domain. Thus, the minimax game takes little effect.![Screenshot 2022-10-16 at 04.51.13](/Users/xinweizhang/Library/Application Support/typora-user-images/Screenshot 2022-10-16 at 04.51.13.png)

Our task now is to design two opposite goals for the adversarial regressor and the feature generator. The goal of the feature generator is to minimize the target disparity or minimize the KL-divergence between the predictions of $f'$ and $f$. The objective of the adversarial regressor is to maximize the target disparity, and we achieve this by minimizing the KL-divergence between the predictions of $f'$ and the ground false predictions of $f$,
$$
L_F(p  ' , p) \triangleq \frac{1} {K} \sum^ K_ k KL(\mathcal{P}_F(\mathcal{J} (p))_k||p  '_ k ),
$$
where $p  ' = (σ ◦ f ' ◦ ψ)(x ^t )$ is the prediction of $f  '$ and $p$ is the prediction of $f$. Compared to directly maximizing the distance from the ground truth predictions of $f$, minimizing $L_F$ can take advantage of the spatial sparsity and effectively change the mean of the output distribution.

##### Overall objectives

The final training objectives are summarized as follows. Though described in different steps, these loss functions are optimized simultaneously in a unified framework.

**Objective 1.** First, we train the generator $ ψ$ and regressor $f$ to detect the source samples correctly. Also, we train the adversarial regressor $f'$ to minimize its disparity with $f$ on the source domain. The objective is as follows:
$$
\min_ {ψ,f,f′} \mathbb{E}_{(x^s,y^s)∼\hat{P}}(L_T((σ ◦ f ◦ ψ)(x^s), y ^s ) + ηL_T((σ ◦ f  ' ◦ ψ)(x^s),(J ◦ f ◦ ψ)(x^s))).
$$
**Objective 2.** Besides, we need the adversarial regressor $f'$ to increase its disparity with $f$ on the target domain by minimizing $L_F$. By maximizing the disparity on the target domain, $f'$ can detect the target samples that deviate far from the support of the source, which can be formalized as follows,
$$
\min_{f'} η\mathbb{E}_{x^t∼\hat{Q}}L_F((σ ◦ f' ◦ ψ)(x^t),(f ◦ ψ)(x^t)).
$$
**Objective 3.** Finally, the generator $ψ$ needs to minimize the disparity between the current regressors $f$ and $f'$ on the target domain.
$$
\min_ ψ η\mathbb{E}_{x^t∼\hat{Q}}ηL_T((σ ◦f' ◦ψ)(x^t),(\mathcal{J} ◦f ◦ψ)(x^t)).
$$


![Screenshot 2022-10-16 at 05.08.17](/Users/xinweizhang/Library/Application Support/typora-user-images/Screenshot 2022-10-16 at 05.08.17.png)
