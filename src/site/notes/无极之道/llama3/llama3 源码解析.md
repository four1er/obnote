---
{"title":"llama3 源码解析","auther":"four1er","created_at":"2025-03-16 20:34","last modify":"2025-03-16 20:33","file path":"无极之道/llama3/llama3 源码解析.md","tags":["llama3"],"dg-publish":true,"permalink":"/无极之道/llama3/llama3 源码解析/","dgPassFrontmatter":true,"created":"2025-03-24T10:39:31.748+08:00","updated":"2025-03-24T10:39:49.166+08:00"}
---


写在前面
最近开始转行做大模型部署推理了，但是光用开源工具部署也不是个事儿，于是这段时间忙里偷闲看了一下 llama 的模型结构，学习记录一下。
本文结构主要从 llama 的模型结构开始，比较 llama 与 transformer 的不同之处，并尝试用代码进行实现，最后结合 llama3 的源码进行分析。
# 模型结构
![image.png](https://gitee.com/four1er/tuchuang/raw/master/img/20250316204710365.png)

如果学习过 Transformer，Llama3 了解起来应该很快，我主要从几个 Llama3 与 transformer 的不同之处开始讲起，主要包括 RMS Norm 归一化、RoPE 位置编码、GQA 注意力机制、FFN-SwiGLU 前馈神经网络。
在每个小节开始，我都会提出一个问题：llama3 要采用这种技术，它与 Transformer 中的不同之处在哪，优缺点是什么？希望在阅读完各个小节之后能对该问题有个答案。
## RMS Norm

> [!question]
> RMS Norm 的优点是什么？为什么要选用 RMS Norm 去做归一化？

RMS Norm (root mean square layer normalization) 是在 Layer Norm 之上做的改进。论文原话的描述是：它仅使用均方根（RMS）统计量将一层神经元的求和输入正则化。与 LayerNorm 相比，RMSNorm 减少了计算量并提高了效率。
简单来说，就是 RMS Norm 没有做 re-center 操作（移除了其中的均值项），所以 RMS Norm 也不是使用整个样本的均值和方差，而是使用平方根的均值来归一化。作者在实验中证明这样做既不会影响正确性，还可以在各个模型上减少约 7%∼64% 的计算时间。
从数学公式上对比 Layer Norm 和 RMS Norm，可以很明显看到其中的差异：
**Layer Normalization**
$$\mu = \frac{1}{n}\sum_{j=1}^{n} x_j$$
$$\sigma = \sqrt{\frac{1}{n}\sum_{j=1}^{n} (x_j - \mu)^2 + \epsilon}$$
$$\text{LayerNorm}(x)_i = \frac{x_i - \mu}{\sigma} \gamma_i + \beta_i$$
其中：$\mu$ 和 $\sigma$ 是输入向量的均值和标准差，$\epsilon$ 是防止除零的小常数（如 10 e−5），$\gamma_i$ 和 ​ $\beta_i$ 是可学习的缩放和偏移参数。
**RMS Normalization**
$$\sigma = \sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2 + \epsilon}$$
$$\text{RMSNorm}(x)_i = \frac{x_i}{\sigma} \gamma_i$$
其中：$\sigma$ 是输入向量的均方根，$\gamma_i$ 是可学习的缩放参数，没有偏移项 $\beta$。

### 代码实现
```python
norm_eps = 1e-5
def rms_norm(tensor, norm_weights):
	return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights[]
```

### 扩展：其他 Norms
- BatchNorm
- LayerNorm
- InstanceNorm
- GroupNorm
具体细节与使用场景在这里不做展开，这里有一张很经典的图：
![image.png](https://gitee.com/four1er/tuchuang/raw/master/img/20250316222659268.png)

感兴趣的同学可以阅读：[一文读懂BN、LN、IN、GN](https://blog.csdn.net/weixin_42127358/article/details/122499267) 。我觉得讲的挺通俗易懂的。
## Rotary Positional Encoding

> [!question]
> RoPE 是什么？对比 transformer 使用的位置编码，RoPE 有什么优缺点？

旋转位置编码（RoPE）是一种能够将相对位置信息依赖集成到 self-attention 中的位置编码方式，但是它确实通过绝对位置编码的方式来实现的。
因为 self-attention 本身是不具备单词位置信息，在计算 QKV 之前需要将位置信息编码到词嵌入向量中。
回顾一下 transformer 是怎么做位置信息编码的呢？Transformer 模型本身只看字和字之间的直接关系（谁重要就和谁互动），但它不知道字的顺序，比如“猫抓老鼠”和“老鼠抓猫”，意思完全相反，所以模型需要知道字的顺序才能区分。这时候就要给每个字加上“座位号”，告诉模型哪个字先出现，哪个字后出现。transformer 中这个座位号用波浪线（正弦函数与余弦函数）表示。
Llama3 采用的旋转位置编码，相当于“用旋转盘给每个字做标记”，它的核心思想是：根据每个字的位置，把它的注意力视角转一个合适的角度，这样计算注意力时，位置信息自然就被融合进去了。具体来说：
1. 给每个字配一个旋转角度：越后面的字，旋转角度越大。比如第 5 个字可能转 30 度，第 10 个字转 60 度。
2. 用旋转矩阵调整向量的方向：在计算注意力时，每个字的向量会根据自己的位置进行“旋转”，这样模型在判断两个字关系时，不仅能看语义是否相关，还能通过旋转角度隐含位置远近的信息。比如两个字距离越远，旋转角度差异越大，内积（相关度）就会更小。

总结一下，transformer 使用的正弦余弦位置编码与 llama3 的旋转位置编码有哪些不同之处呢？
正弦余弦位置编码：无额外训练参数，计算速度快，支持序列长度有限，适合短文本任务（如机器翻译、文本分类），对计算资源敏感的场景。
旋转位置编码：包含相对位置信息，支持长序列上下文，计算复杂度高，需要额外调参，适合长文本生成、强外推能力的任务（如超长上下文对话）等。
### 数学实现
推导细节可以参看：[博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)，作者是 RoPE 的发明者。
简单点来说，就是对于输入的 q、k 我们需要对它做一些变换，让它变换之后的结果携带绝对位置信息。额外地，由于 attention 的核心运算是内积，所以我们希望的内积的结果带有相对位置信息。
从这个目标出发，为 q、k 设计操作：$\boldsymbol{f}(\cdot, m),\boldsymbol{f}(\cdot, n)$，使得：
$$
\begin{equation}\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)\end{equation}
$$
然后一通推导之后可以得到：$$
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)}
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}\end{equation}
$$
>  这里怎么推导出来的哥们确实没看懂...

这个也可以写成：$$
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}\end{equation}
$$
从二维扩展到任意偶数维的 RoPE，我们都可以表示为二维情形的拼接：
$$
\begin{equation}\scriptsize{\underbrace{\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}\end{equation}
$$
最后我们可以通过以下公式实现：
$$
\begin{aligned}
\begin{equation}\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}\end{equation}
\end{aligned}
$$
其中：$m$ 为位置索引，$q_0, q_1,…,q_{d-1}$ 为第 $m$ 个 token 的向量表示。$\theta_i$ 的计算方式为：$\theta_i=\frac{1}{10000^\frac{2i}{d}}$，其中 $d$ 为词向量的维度。
具体来说，我们需要先预计算出 $\theta_i$ 的所有可能的取值，然后对于不同位置 m 的旋转角度可以记为 $\theta_m=m\cdot \theta$，那么对于一个二维向量 $(x_m, y_m)$，其复数形式为 $z_m=x_m+i \cdot y_m$，在旋转之后的向量可以被计算为：$$z_m'=z_m \cdot e^{i\cdot\theta_m}=(x_mcos⁡\theta _m−y_m sin⁡\theta_m)+i(x_m sin⁡\theta_m+y_m cos⁡\theta_m)$$
扩展到高位向量的时候，我们记向量维度为 $d$， $d$ 通常为偶数。RoPE 将向量划分为 $d/2$ 组二维子空间，每组独立旋转。第 $k$ 组的角度为 $\theta_{m,k}=m \cdot \theta^{2*k/d}$, 其中 $\theta$ 通常取 $10^4$, 以控制旋转速度。

### 代码实现

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # 计算频率参数（θ_i）
        # inv_freq: (dim//2, )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, x, seq_len):
    	# t: (seq_len, )
        t = torch.arange(seq_len).type_as(self.inv_freq)
        # 生成旋转角度：θ_i * m
        # freqs: (seq_len, dim//2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 将 freqs 在最后一个维度拼接两次,扩展为复数形式（cosθ + i sinθ）
        # emb: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_heads, head_dim]
        """
        batch, seq_len, _, _ = x.shape
        cos, sin = self._compute_cos_sin(x, seq_len)
        
        # 扩展维度用于广播
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
        
        # 旋转操作
        rotated = (x * cos) + (self.rotate_half(x) * sin)
        return rotated
```

测试代码
```python
if __name__ == "__main__":
    dim = 128
    seq_len = 512
    batch_size = 2
    num_heads = 8

    embeddings = torch.randn(batch_size, seq_len, num_heads, dim)
    rope = RotaryEmbedding(dim)
    rotated_embeddings = rope(embeddings)
    
    print("input:", embeddings.shape)
    print("output:", rotated_embeddings.shape)
```

## Grouped-MultiQuery-Attention
Llama3 采用了分组查询注意力机制（简称 GQA），对比起 transformer 使用的多头注意力机制有什么不同之处呢？
我们先简单回顾一下什么是注意力，简单点说，就是对于一个给定的词语，我怎么找到哪些词是跟这个目标词相关的？哪些词的含义会对我的目标词产生影响？大佬们用了一个很巧妙的方法，就像从淘宝搜东西一样，输入一个 query，根据这个 query 从数据库里面捞数据，那怎么判断候选数据是否应该被推荐出来呢？那这里就需要去计算 query 与数据库中的 key 的相似度了，相似度高的，那自然会被推荐到首位。只不过在 attention 的计算中，不会因为某个候选项不相干就直接干掉，取而代之的，是给他一个很小的权重，用这个很小的权重去乘以 value 就可以啦。
我们再用白话简单描述一下 attention 的计算公式：$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V
$$
Q、K、V 作为矩阵，可以高效的批量计算。对于每个 Q、K、V，它们的每一行都可以理解成是对原 token 的一个向量表示，$QK^\top$ 就是对上面"query 与候选项关联度"的计算，计算结果的每一行可以理解成当前 token 与其他 token 的关联度如何。然后对这个关联度结果做一些二次加工，最后乘上 V 就大功告成了。
讲完了 attention 之后，来复习一下 transformer 使用的多头注意力机制，多头注意力机制是在自注意力的基础上增加多个头来并行的处理输入信息，可以简单理解成不同注意力头对语句的关注点不同（比如有的关注因果关系、有的关注修饰词句等），这样多个头就可以捕捉到更丰富的上下文语义。
多头注意力的基本计算逻辑如下：
1. 线性变换：首先，对输入序列中的每个位置的向量分别进行三次线性变换（即加权和偏置），生成查询矩阵 Q, 键矩阵 K, 和值矩阵 V。在多头注意力中，这一步骤实际上会进行 h 次（h 为头数），每个头拥有独立的权重矩阵，从而将输入向量分割到 h 个不同的子空间。
2. 并行注意力计算：对每个子空间，应用自注意力机制计算注意力权重，并据此加权求和值矩阵 V，得到每个头的输出。
3. 合并与最终变换：将所有头的输出拼接起来，再经过一个最终的线性变换和层归一化，得到多头注意力的输出。
多头注意力机制很牛，但是问题出在它需要的参数量太大了，在显存寸土寸金的当下，怎么既能减少参数量节省显存，同时又能保证模型精度不受影响就成了一个值得研究的话题。
### 注意力机制对比
大模型常用的注意力机制包括：MHA (Multi-Head Attention)、MQA (Multi-Query Attention)、GQA (Grouped-query Attention)。
从直观上来看，三种注意力对比如下：
![image.png](https://gitee.com/four1er/tuchuang/raw/master/img/20250318004126874.png)

可以看到它们主要的区别在于 key、value 的参数量不同，这其实代表了精度与显存之间的 trade-off。
从参数量大小进行对比：
设假设输入的维度是 $d_{model}$，注意力头数为 $h$，每个头的维度是 $d_k$（$d_k$ = $\frac{d_{model}}{h}$，比如 $d_{model}=4096$，$h=32$，那么 $d_k=128$）。
- MHA 中每个注意力头有它自己的 Q、K、V，并且所有注意力头的 Key 和 Value 矩阵权重不共享。对于每个头的 Q、K、V 的参数矩阵都是 $d_{model}*d_k$, 那么每个头就是 $3*d_{model}*d_k$, h 个注意力头是 $h*3*d_{model}*d_k$,再加上输出矩阵 $W_o$ 的大小也是 $d_{model}^2$, 所以总参数量为：
$$
d_{model}^2 + h*3*d_{model}*d_k = d_{model}^2 + h*3*d_{model}*\frac{d_{model}}{h}=4*d_{model}^2
$$
- MQA 中所有头共享同一组 K 和 V 矩阵，仅 Q 独立，所以 Q 的参数量是 $h*d_{model}*d_k=d_{model}^2$, 而 K 和 V 的参数量为：$2*d_{model}*\frac{d_{model}}{h} = 2*\frac{d_{model}^2}{h}$ ,再加上输出矩阵 $W_o$ 的大小是 $d_{model}^2$，可以算出总的参数量为：$$
2*d_{model}^2 + 2*\frac{d_{model}^2}{h}
$$
- GQA 中将头分组（分成 $g$ 组），组内共享 K/V 矩阵。所以 Q 的参数量不变还是 $d_{model}^2$，K/V 的组共享参数为：$2*g*\frac{d_{model}^2}{h}$, 最后加上输出矩阵 $W_o$ 的大小 $d_{model}^2$，可以算出总参数量为：$$
d_{model}^2+2*g*\frac{d_{model}^2}{h}+d_{model}^2 = 2*d_{model}^2 + 2*g*\frac{d_{model}^2}{h}
$$
以 Llama3-8B 的参数计算（$d_{model}=4096$, $h=32$, $g=4$）可以计算出：
MHA 需要 67108864, MQA 需要 34603008（相对 MHA 减少 48.4%）, GQA 需要：37748736（相对 MHA 减少 43.75%）
从模型质量上来说，MHA 性能最优，但资源消耗大；MQA 推理效率高，但可能损失部分精度；GQA 在参数量与性能间平衡，可以通过调整 g 的大小来灵活的实现平衡。
### 实现
在接下来的代码实现中，$d_{model}$ 表示隐藏层的维度，$d_k$ 表示 head 的维度。
我们从 transformer 的 MultiHeadAttention 开始：
1. 首先需要准备好 Q、K、V 的权重矩阵，它们的 shape 都是 $d_{model}*d_{model}$
2. 对 Q、K、V 按 head 数进行分割，分割之后的维度是 $d_{model} * d_{head} * d_{dk}$
3. 计算 attention：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$，记得设置掩码
4. 拼接 head 的输出，维度重新变为 $d_{model}*d_{model}$
具体实现如下：

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads = 32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads 
        
        self.W_q = nn.Linear(d_model, d_model) # query (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) # key (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model) # value (d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # output (d_model, d_model)
        
        self.scale = torch.sqrt(torch.tensor(self.head_dim)) # scale: sqrt(head_dim)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, mask=None):
        """
        input x: (batch_size, num_heads, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """
        
        Q = self.split_heads(self.W_q(x))      # (batch_size, num_heads, seq_len, head_dim)
        K = self.split_heads(self.W_k(x))      # (batch_size, num_heads, seq_len, head_dim)
        V = self.split_heads(self.W_v(x))      # (batch_size, num_heads, seq_len, head_dim)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
        	mask = mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # 将掩码位置设为负无穷
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_probs, V)     # (batch_size, num_heads, seq_len, head_dim)
        
        # connact heads outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            x.size(0), -1, self.d_model
        )  # (batch_size, seq_len, d_model)
        
        output = self.W_o(attn_output)
        return output
```

GQA 的实现与 MHA 类似，区别之处在于对 K、V 分了多个组，组间共享 KV。
```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0  
        
        self.W_q = nn.Linear(d_model, d_model)  # query (d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim * num_groups) # key (d_model, head_dim * num_groups)
        self.W_v = nn.Linear(d_model, self.head_dim * num_groups) # value (d_model, head_dim * num_groups)
        self.W_o = nn.Linear(d_model, d_model)  # output (d_model, d_model)
        
        self.kv_heads = num_groups
        self.group_size = num_heads // num_groups  # 每组包含的查询头数
        
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len, seq_len)
        output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.W_k(x)  # (batch_size, seq_len, head_dim * num_groups)
        v = self.W_v(x)  # (batch_size, seq_len, head_dim * num_groups)
        
        # reshape q, k, v
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # q shape: (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)   # k shape: (batch_size, kv_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)   # v shape: (batch_size, kv_heads, seq_len, head_dim)
        
        # 组内复制键值（实现分组共享）
        if self.group_size > 1:
            # 在头维度上复制group_size次 (batch_size, kv_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
            k = k.repeat(1, self.group_size, 1, 1)  # (batch_size, num_heads, seq_len, head_dim)
            v = v.repeat(1, self.group_size, 1, 1)  # (batch_size, num_heads, seq_len, head_dim)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # attn_scores: (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # connact heads output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_o(attn_output)  # (batch_size, seq_len, d_model)
        return attn_output

```

## FF-SwiGLU
Llama 3 采用的 FF-SwiGLU（FeedForward-SwiGLU）前馈网络是对传统 Transformer 中 FFN（Feed-Forward Network）模块的改进，其核心是结合了 Swish 激活函数和 GLU（Gated Linear Unit）门控机制。
### SwiGLU 激活函数
在介绍 Swish 之前，还是先简单回顾一下常用的激活函数 ReLU。
ReLU 的数学实现很简单，公式为：$f(x) = max(0, x)$。它因快速的 SGD 收敛速度和较低的计算复杂度而备受青睐，但是 ReLU 有个无法绕过的问题，就是神经元坏死，这会导致某些神经元可能永远不会被激活，进而导致相应参数一直得不到更新。

> 什么是神经元坏死：当 ReLU 神经元的输入为负值并长时间保持负值，那么这个神经元之后的梯度就永远是 0 了，也就是 ReLU 神经元坏死，不再对任何数据有所响应。
> 产生该问题主要原因包括参数初始化设置以及学习率设置过大。

Swish 是一种非线性的激活函数，它的数学表达为：$Swish_\beta(x) = x \ast sigmoid(\beta x)$, 其中 $\beta$ 经常被设置为 1，那么这个函数就简化成了：$$Swish_1(x) = x \ast sigmoid(x)$$ 其实也就变成了 $SiLU(x)$ 激活函数。
Swish 函数的形状很类似 ReLU，但是也有些区别，比如负区间输出非零但逐渐衰减，正区间则类似 ReLU；并且由于导数包含 Sigmoid 项，因此也可以避免梯度消失的问题
对比起 ReLU，Swish 具有更平滑的梯度，可以确保神经网络中的神经元继续产生输出，从而缓解梯度消失问题，因此更适用于深层网络。但是成也萧何败萧何，由于引入了 $sigmoid$，Swish 的计算复杂度要明显高于 ReLU。
从直观上来看，他们的曲线对比：
![](https://gitee.com/four1er/tuchuang/raw/master/img/20250323014444622.png)

GLU 是一种用于增强模型表现的激活函数，它通过引入门控机制，使得模型能够选择性地通过信息，从而提高模型的表达能力和性能。它的数学公式为：$GLU(x)=(W*x+b)⊗sigmoid(V*x+c)$。公式很吓人，但是其实很简单，就是对于一个输入 x，它引入了两个线性变换，其中一个作为激活函数，来控制（通过加权实现）另一个的激活状态。
有了上面两个概念之后，理解 SwiGLU 就很简单了，其实它就是将 GLU 中的激活函数换成了 Swish 激活函数，数学表达 (忽略偏置项) 为：$$ SwiGLU(x, W, V) = Swish_1(x ∗ W) ⊗ (x ∗ V) $$
在理解了 SwiGLU 之后，我们参考 transformer 的前馈神经网络结构：$FFN(x,W_1,W_2) = max(0, xW_1 + b_1)W_2+b_2$
可以构建基于 SwiGLU 的前馈神经网络：
$$

FF-SwiGLU(x,W_1, W_2, V) = (Swish_1(xW_1)⊗xV)W2

$$
可以看到，其实 FF-SwiGLU 是要比 FFN 要多了一个权重参数的，而前馈神经网络的参数大小其实是占了总模型参数大小的主要部分，那有没有方法能减少这个额外权重参数带来的影响呢？有，Llama 通过调整前馈神经网络中隐藏层的维度大小解决了这个问题。
### 实现
Python 实现比较简单，对着公式堆积木就行
```python
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    
    def forward(self, x):
        x1 = F.linear(x, self.w1.weight)
        x2 = F.linear(x, self.w2.weight)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, self.w3.weight)
```

### FF-SwiGLU 与 FFN 的参数量对比
在 transformer 中的 FFN 有两个 MLP 层，这两个 MLP 层的参数量分别为：$d_{model}*4d_{model}$ 和 $4d_{model}×h_{model}$，总的参数量为 $8d_{model}^2$。
而 FF-SwiGLU 的实现为：$FFNSwiGLU(x,W_1, W_2, V) = (Swish_1(xW_1)⊗xV)W2$，其中 $W_1$ 和 $V$ 的维度相同，记为 ($d_{model}$, $a*d_{model}$), 作用是为了升维，而 $W_2$ 的作用是降维，将结果还原到输入的维度大小，大小为 $(a*d_{model}, d_{model})$。
为了保证参数量不动，就有了这么一个方程式：$$
8 d_{model}^2 = 3*a*d_{model}^2
$$
可以解出 $a=\frac{8}{3}$, 这个值乘以模型维度有可能除不尽，但是可以作为预估值。
在 Llama3 8B 的模型中，隐藏层维度大小是：4096，乘上 $a=\frac{8}{3}$，结果约是 10992。实际实现中还乘了一个缩放参数 1.3，$10992 * 1.3 = 14289$ ，与真实值：14336 相近。
源码中的 8/3 也是这么来的：
```python
// ...
class TransformerBlock(nn.Moduel):
	self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
// ...

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        dim: 输入的特征维度 4096
        hidden_dim: 隐藏层的特征维度 4096*4
        multiple_of:  1024
        ffn_dim_multiplier: 1.3
        """
        super().__init__()
        # hidden_dim = 2 * hidden_dim / 3 = 2 * 4 * dim / 3 = 8/3 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        // ....
```

### 参数量的计算
以 LLM-Research/Meta-Llama-3.1-8B-Instruct 为例，参数配置为：
```json
{
   "dim":4096,
   "n_layers":32,
   "n_heads":32,
   "n_kv_heads":8,
   "vocab_size":128256,
   "ffn_dim_multiplier":1.3,
   "multiple_of":1024,
   "norm_eps":1e-05,
   "rope_theta":500000.0,
   "use_scaled_rope":true
}
```
我们来算算，为什么有 8B 级别的参数。
首先算嵌入层与输出层参数：
- 嵌入层：词表大小 `vocab_size=128256` 与隐藏维度 `dim=4096` 的乘积：$128256 * 4096 = 525,336,576$
- 输出层：与嵌入层大小一致，大小为：$525,336,576$
嵌入层与输出层的参数总大小为：$$
525,336,576 + 525,336,576 = 1,050,673,152
$$

重点是隐藏层 (transformer 层)：
- GQA 的参数大小，带入上面的公式 $2*d_{model}^2 + 2*g*\frac{d_{model}^2}{h}$，可以算出大小为：$2 * 4096^2 + 2 * 4 * \frac{4096^2}{32} = 37,748,736$
- FFN-Swish 层的参数大小：ffn 层的中间维度会进行一次额外的缩放（ffn_dim_multiplier），大小是 $\frac{4096 * 8/3 * 1.3}{1024} * 1024 = 14336$, 所以总参数大小为 $3 * 14336 * 4096 = 176,160,768$
- RMSNorm 层的参数大小：2 个归一化层，每个归一化层是 4096，所以总参数大小是 8192
总共有 32 层 transformer 层参数总大小为：$$
32 * (37,748,736 + 176, 160,768 + 8192) = 6,845,366,272
$$
将两个值相加，可以得到总参数量约等于：7896039424，约等于 8B 了。

### 小结
至此，Llama3 的核心模块都已经实现完毕，源码看起来应该就很快了，10 分钟快速带过。主要是对现有组件的拼接，俗称搭积木。

# Llama3 源码解读
Llama3 的源码部分，我会从上往下进行分析，先看大体的模型结构，然后分析每个结构的实现。
## 模型参数配置
首先看模型参数的配置：
```python
@dataclass
class ModelArgs:
    dim: int = 4096 # model dimension
    n_layers: int = 32 # number of layers
    n_heads: int = 32 # number of heads
    n_kv_heads: Optional[int] = None # number of key/value heads
    vocab_size: int = -1 # vocab size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None # feedforward dimension multiplier
    norm_eps: float = 1e-5 # normalization epsilon, default is 1e-5
    rope_theta: float = 500000 # RoPE theta

    max_batch_size: int = 32 # maximum batch size
    max_seq_len: int = 2048 # maximum sequence length
```
这个配置会被实际的模型参数所覆盖，还是以 LLM-Research/Meta-Llama-3.1-8B-Instruct 来说，实际的模型参数如下：
```json
{
   "dim":4096,
   "n_layers":32,
   "n_heads":32,
   "n_kv_heads":8,
   "vocab_size":128256,
   "ffn_dim_multiplier":1.3,
   "multiple_of":1024,
   "norm_eps":1e-05,
   "rope_theta":500000.0,
   "use_scaled_rope":true
}
```
可以看到 Llama3 8B 使用的隐藏层模型维度为 4096，有 32 层个隐藏层和 32 个注意力头，而 kv_heads 为 8 表示将 32 个查询头分为 8 组，每组 4 个查询头共享 1 个 KV 头（共 8 个 KV 头）。

## Transformer 构成
接下来先看整体的 transformer 结构，其中涉及到的 RMSNorm、freqs_cis 和 TransformerBlock 我们会具体分析：
```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        # n_layers: number of layers， model config: 32
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        # 根据n_layers的数量创建数个TransformerBlock
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # norm 层使用RMSNorm
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # 预先计算好RoPE中的频率值，并用复数形式表达
        # 传入的参数分别是：
        # - dim: 模型的维度，这里传入的是每个头的维度
        # - max_seq_len: 最大序列长度，这里传入的是模型的最大序列长度*2，即预留好充足的buffer
        # - rope_theta: RoPE的缩放因子，模型参数中给的是 50000
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens: [batch_size, seq_len]
        _bsz, seqlen = tokens.shape
        # h: [batch_size, seq_len, dim]
        h = self.tok_embeddings(tokens)
        
        # pre-compute RoPE freqs
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 设置掩码
        mask = None
        if seqlen > 1:
            # mask: [seq_len, seq_len]， value: -inf
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            # 对角线以上的元素保留，对角线及以下的元素被置为 0
            # 目的是防止当前token能看到未来的token
            mask = torch.triu(mask, diagonal=1)

            # 设置 kv cache mask
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
            # mask: [seq_len, cache_len + seq_len]

        # TransformerBlock.forword()
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

```
这里对掩码的设置稍作讲解：
假设当前序列长度 `seqlen = 3`, 缓存长度 `start_pos = 2`, 那么对于当前序列中的第一个 token，它应当能关注到它自身以及前 2 个计算过的 token，对于后 2 个 token 是没办法关注到的；同样，对于第二个 token，它能关注到前 3 个 token + 自身，最后一个 token 没办法关注到。实际数据如下：
![image.png](https://gitee.com/four1er/tuchuang/raw/master/img/20250323153428500.png)

首先是 RMSNorm, RMSNorm 的实现没什么好说的，就是上面的数学公式。
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

然后是旋转位置编码 RoPE：
```python
import torch

def precompute_freqs_cis(dim: int, end: int, constant: float = 10000.0):
    '''
    计算cos和sin的值，cos值在实部，sin值在虚部，类似于 cosx+j*sinx
    :param dim: q,k,v的最后一维，一般为emb_dim/head_num
    :param end: 句长length
    :param constant： 这里指10000
    :return:
    复数计算 torch.polar(a, t)输出， a*(cos(t)+j*sin(t))
    '''
    # freqs: 计算 1/(10000^(2i/d) )，将结果作为参数theta
    # 形式化为 [theta_0, theta_1, ..., theta_(d/2-1)]
    freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # [d/2]

    # 计算m
    t = torch.arange(end, device=freqs.device)  # [length]
    # 计算m*theta
    freqs = torch.outer(t, freqs).float()  # [length, d/2]
    # freqs形式化为 [m*theta_0, m*theta_1, ..., m*theta_(d/2-1)],其中 m=0,1,...,length-1

    # 计算cos(m*theta)+j*sin(m*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # freqs_cis: [cos(m*theta_0)+j*sin(m*theta_0),  cos(m*theta_1)+j*sin(m*theta_1),), ..., cos(m*theta_(d/2-1))+j*sin(m*theta_(d/2-1))]
    # 其中j为虚数单位， m=0,1,...,length-1
    return freqs_cis # [length, d/2]

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # (1, length, 1, d/2)
    return freqs_cis.view(*shape) # [1, length, 1, d/2]

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,):
    # 先将xq维度变为[bs, length, head,  d/2, 2], 利用torch.view_as_complex转变为复数
    # xq:[q0, q1, .., q(d-1)] 转变为 xq_: [q0+j*q1, q2+j*q3, ..., q(d-2)+j*q(d-1)]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # [bs, length, head, d/2]
    # 同样的，xk_:[k0+j*k1, k2+j*k3, ..., k(d-2)+j*k(d-1)]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, length, 1, d/2]
    # 下式xq_ * freqs_cis形式化输出，以第一个为例, 如下
    # (q0+j*q1)(cos(m*theta_0)+j*sin(m*theta_0)) = q0*cos(m*theta_0)-q1*sin(m*theta_0) + j*(q1*cos(m*theta_0)+q0*sin(m*theta_0))
    # 上式的实部为q0*cos(m*theta_0)-q1*sin(m*theta_0)，虚部为q1*cos(m*theta_0)+q0*sin(m*theta_0)
    # 然后通过torch.view_as_real函数，取出实部和虚部，维度由[bs, length, head, d/2]变为[bs, length, head, d/2, 2]，最后一维放实部与虚部
    # 最后经flatten函数将维度拉平，即[bs, length, head, d]
    # 此时xq_out形式化为 [实部0，虚部0，实部1，虚部1，..., 实部(d/2-1), 虚部(d/2-1)]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # [bs, length, head, d]
    # 即为新生成的q

    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

## TransformerBlock
TransformerBlock 就是 LLama3 的结构图中最复杂的那一块了，跟 Transformer 类似，由 GQA attention 和 FF-SwiGLU 构成，通过残差进行连接。
![image.png](https://gitee.com/four1er/tuchuang/raw/master/img/20250323175514184.png)
```python
# TransformerBlock 中包含了注意力机制和前馈神经网络
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        # 初始化attention层
        self.attention = Attention(args)
        # 初始化ffn层
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # 在计算attention 和 ffn的时候需要记得把结果与输入相加以得到最终输出（即残差连接）
        # 计算attention前需要对输入进行norm处理
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # 将attention的输出进行norm处理，然后输入到ffn层。
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

GQA 的实现与前文的实现类似，对分组后的组内 KV 进行复制扩散，不过在源码中的实现额外增加了 KV Cache 的维护，每次计算时增量更新 KV Cache。
```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # n_kv_heads 被设置为 8
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 获取并行化设备数量，我们假设只有一个设备，便于理解
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 单台设备上需要处理的头的数量，也就是 32
        self.n_local_heads = args.n_heads // model_parallel_size
        # 单台设备上需要处理的 kv 头的数量，也就是 8
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 每个头的重复次数 32/8 = 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度 4096/32 = 128
        self.head_dim = args.dim // args.n_heads

        # wq [dim] -> [n_heads * head_dim]
        self.wq = init(...)
        # wk [dim] -> [n_kv_heads * head_dim]
        self.wk = init(...)
        # wv [dim] -> [n_kv_heads * head_dim]
        self.wv = init(...)
        # wo [n_heads * head_dim] -> [dim]
        self.wo = init(...)

        # init kv cache
        self.cache_k = init()
        self.cache_v = init()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # x [bsz, seqlen, dim]
        bsz, seqlen, _ = x.shape
        
        # 对输入进行线性变换，生成 query、key、value
        # xq: [bsz, seqlen, n_local_heads, head_dim]
        # xk: [bsz, seqlen, n_local_kv_heads, head_dim]
        # xv: [bsz, seqlen, n_local_kv_heads, head_dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 对q和k应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k: [max_batch_size, max_seq_len, n_local_kv_heads, head_dim]
        # self.cache_v: [max_batch_size, max_seq_len, n_local_kv_heads, head_dim]
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        # 更新kv缓存
        # xk: [bsz, seqlen, n_local_kv_heads, head_dim]
        # xv: [bsz, seqlen, n_local_kv_heads, head_dim]
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        # 从缓存里面获取kv
        # keys: [bsz, start_pos + seqlen, n_local_kv_heads, head_dim]
        # values: [bsz, start_pos + seqlen, n_local_kv_heads, head_dim]
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # start_pos = cache_len
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

FF-SwiGLU 的实现
```python
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        dim: 输入的特征维度 4096
        hidden_dim: 隐藏层的特征维度 4096*4
        multiple_of: 确保隐藏层的维度是multiple_of的倍数，值为：1024
        ffn_dim_multiplier: 1.3
        """
        super().__init__()
        # hidden_dim = 2 * hidden_dim / 3 = 2 * 4 * dim / 3 = 8/3 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # swish_glu math implementation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

# Ref
1. [llama3 github](https://github.com/meta-llama/llama3)
2. [llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)
3. Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." *Advances in Neural Information Processing Systems* 32 (2019).
4. Wu, Yuxin, and Kaiming He. "Group normalization." *Proceedings of the European conference on computer vision (ECCV)*. 2018.
5. https://kazemnejad.com/blog/transformer_architecture_positional_encoding
6. Transformer 中的位置编码 https://0809zheng.github.io/2022/07/01/posencode.html
7. [大模型系列：快速通俗理解Transformer旋转位置编码RoPE](https://blog.csdn.net/2401_84495872/article/details/139698878)
8. [常用激活函数对比](https://www.cnblogs.com/Joejwu/p/Joejwu_blog210618.html)
9. [llama3架构及源码解析](https://mp.weixin.qq.com/s?__biz=Mzk0NzY4NDYyOA==&mid=2247484598&idx=1&sn=540b97d2edd91496fcee1421ab693df8&chksm=c2402aa2648d9fb7bbd7bb1e4e0550435300969c5c27df574e4c9dd4085a246c631b56dded90#rd)
10. Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Swish: a self-gated activation function." _arXiv preprint arXiv:1710.05941_ 7.1 (2017): 5.
11. Dauphin, Yann N., et al. "Language modeling with gated convolutional networks." _International conference on machine learning_. PMLR, 2017.
