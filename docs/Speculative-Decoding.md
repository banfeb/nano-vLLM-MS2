### Speculative-Decoding

原论文：[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

#### 调用链


#### 一个例子
##### 0. 场景设定

当前前缀是：
$$
x = \text{“The capital of France is”}
$$
设 draft model（小模型）准备提议 $K=3$ 个 token。

小模型提议的草稿是：
$$
\hat{x}_{t+1}=\text{Paris},\quad
\hat{x}_{t+2}=\text{.},\quad
\hat{x}_{t+3}=\text{beautiful}
$$
也就是它草拟出一句：

> The capital of France is Paris . beautiful

当然最后这个句子不一定合理，这正是后面要验证的。

------

##### 1. 第一个位置：接受还是拒绝

先看第一个位置，也就是在前缀
$$
\text{“The capital of France is”}
$$
后面，下一个 token 是什么。

假设此时：

###### 小模型分布 $q_1$

- Paris: 0.60
- London: 0.20
- Lyon: 0.10
- 其他: 0.10

###### 大模型分布 $p_1$

- Paris: 0.90
- London: 0.03
- Lyon: 0.02
- 其他: 0.05

小模型实际提议的是 `Paris`。

接受概率是：
$$
\alpha_1 = \min\left(1,\frac{p_1(\text{Paris})}{q_1(\text{Paris})}\right)
= \min(1,\frac{0.90}{0.60}) = 1
$$
所以：

- 第 1 个 token **一定接受**

于是当前确认输出变成：

> The capital of France is Paris

------

##### 2. 第二个位置：继续接受还是拒绝

现在看第二个位置。此时上下文已经变成：
$$
\text{“The capital of France is Paris”}
$$
小模型提议第 2 个 token 是 `.`。

假设：

###### 小模型分布 $q_2$

- . : 0.70
- and: 0.10
- , : 0.10
- is: 0.05
- 其他: 0.05

###### 大模型分布 $p_2$

- . : 0.56
- and: 0.14
- , : 0.10
- is: 0.10
- 其他: 0.10

接受概率：
$$
\alpha_2 = \min\left(1,\frac{p_2(.)}{q_2(.)}\right)
= \min(1,\frac{0.56}{0.70})=0.8
$$
这表示：

- 第 2 个 token `.` **不是必然接受**
- 而是以 0.8 的概率接受

现在假设我们采样到“接受”。

于是第 2 个 token 也通过了。

当前确认输出变成：

> The capital of France is Paris .

------

##### 3. 第三个位置：首次拒绝

现在看第 3 个位置。当前上下文是：
$$
\text{“The capital of France is Paris .”}
$$
小模型提议的 token 是 `beautiful`。

假设：

###### 小模型分布 $q_3$

- beautiful: 0.50
- It: 0.20
- The: 0.10
- <eos>: 0.10
- 其他: 0.10

###### 大模型分布 $p_3$

- beautiful: 0.05
- It: 0.40
- The: 0.20
- <eos>: 0.25
- 其他: 0.10

注意这里，小模型很偏爱 `beautiful`，但大模型并不认同。

接受概率：
$$
\alpha_3 = \min\left(1,\frac{p_3(\text{beautiful})}{q_3(\text{beautiful})}\right)
= \min(1,\frac{0.05}{0.50}) = 0.1
$$
也就是说：

- `beautiful` 只有 10% 概率被接受

现在假设这次随机结果是：**拒绝**。

于是：

- 前两个 token 接受
- 第三个 token 第一次被拒绝

所以小模型那段草稿里，真正保留下来的只有：
$$
[\text{Paris}, \text{.}]
$$

------

##### 4. 为什么不能直接从大模型 $p_3$ 重新采样

很多人第一次看会想：

> 既然 `beautiful` 被拒绝了，那直接从大模型分布 $p_3$ 采一个新 token 不就行了吗？

**不行。**

因为我们刚刚已经用“接受/拒绝”这个动作，对 `beautiful` 做过一次筛选了。
 这一步已经改变了概率质量的分配。

更准确地说：

- 小模型先“提议了” `beautiful`
- 然后我们又拒绝了它

所以现在不能假装什么都没发生，再直接从 $p_3$ 采样。
 否则最终分布就会偏掉，不再严格等于大模型原始分布。

所以这时要用一个**修正分布**。

------

##### 5. 修正分布怎么构造

论文里的核心想法是：

在拒绝位置，不从 $p$ 直接采样，而是从“剩余概率质量”里采样。

一种最常见的写法是：
$$
r(x) \propto \max(0,\, p(x)-q(x))
$$
更准确地说，是把所有候选 token 的
$$
\max(0,p(x)-q(x))
$$
算出来，再归一化，形成一个新的概率分布 $r$。

------

##### 6. 在这个例子里实际算一遍修正分布

我们看第 3 个位置的几个候选 token：

| token     | $p_3(x)$ | $q_3(x)$ | $p_3(x)-q_3(x)$ | $\max(0,p_3-q_3)$ |
| --------- | -------- | -------- | --------------- | ----------------- |
| beautiful | 0.05     | 0.50     | -0.45           | 0                 |
| It        | 0.40     | 0.20     | 0.20            | 0.20              |
| The       | 0.20     | 0.10     | 0.10            | 0.10              |
| <eos>     | 0.25     | 0.10     | 0.15            | 0.15              |
| 其他      | 0.10     | 0.10     | 0               | 0                 |

把正的部分加起来：
$$
0.20 + 0.10 + 0.15 = 0.45
$$
所以修正分布 $r$ 是：

- $r(\text{It}) = 0.20/0.45 \approx 0.444$
- $r(\text{The}) = 0.10/0.45 \approx 0.222$
- $r(\text{}) = 0.15/0.45 \approx 0.333$
- $r(\text{beautiful}) = 0$

你看，`beautiful` 已经被拒绝了，所以在修正分布里它直接没有机会了。

------

##### 7. 用修正分布补 1 个 token

现在从修正分布 $r$ 里采样一个 token。

假设这次采样结果是：
$$
\text{<eos>}
$$
那么这一轮 speculative sampling 的最终产出就是：

- 接受的前缀：`Paris .`
- 拒绝后修正补的 token：`<eos>`

所以最终这轮输出的是：
$$
[\text{Paris}, \text{.}, \text{<eos>}]
$$
也就是：

> The capital of France is Paris.

这轮就结束了。
