---
{"title":"Raft","auther":"four1er","created_at":"2025-01-17 11:45","last modify":"2025-01-17 11:45","file path":"Papers/Raft.md","tags":["distributed_sytem","Raft","papers"],"dg-publish":true,"permalink":"/Papers/Distributed System/Raft/","dgPassFrontmatter":true,"created":"2025-02-06T10:50:22.640+08:00","updated":"2025-02-10T11:09:22.789+08:00"}
---

# 分布式共识算法背景
## 分布式系统数据多副本
对于包含成千上百个节点的分布式系统，节点宕机是家常便饭的事情，系统必须保证这种情况下的可用性。也就是常说的 " 高可用 "
对于**无状态系统**，直接部署多台机器即可，再配以负载均衡手段即可。
对于**有状态系统**，例如：数据库、文件存储服务、消息队列等，通常采用数据多副本的形式实现高可用。
随之而来的问题：分布式数据多副本的一致性。
## 传统主从架构的问题
以典型的一主多从的数据库架构为例，常用的同步方式有两种：1. 同步复制 2. 半同步复制 3. 异步复制。
### 1. 同步复制
写请求发给主库，主库同步更新到其他副本后才返回。
这种方式可以保证副本间的强一致性，写成功返回后，从任意副本读到的数据都是一致的。
缺点是可用性很差，只要任意一个副本写失败，写请求将执行失败
### 2. 半同步复制
当主库执行完一条事务并写入 Binlog 后，不会立即返回给客户端，而是会等待至少一个从库确认已经接收到这个 Binlog。只有在收到从库的确认信号后，主库才会返回事务提交成功的响应给客户端。如果在设定的超时时间内没有收到从库的确认，主库会回退到异步复制模式，以保证系统的可用性。
优点：至少有一个从库已经确认了该事务的 binlog，可以减少数据丢失的风险。
缺点：等待确认的开销，另外如果网络不好可能导致频繁超时。
### 3. 异步复制
写请求首先发送给主库，主库写成功后立即返回成功，然后异步更新其他副本。
优点：可用性好。主库写成功就成功。
缺点：不能保证数据的一致性。1. 读写不一致。2. 主库宕机会导致数据丢失。

> [!summary]
> 传统的主从复制无法同时保证数据的一致性与可用性。

# Raft
https://raft.github.io/
## Abstract
Raft 将一致性算法分解成了几个关键模块，例如领导人选举、日志复制和安全性。同时 Raft 还设计了新的机制来运行集群成员的动态变更。
## Introduction
Raft 算法相比其他共识算法的特性：
- 强领导人：和其他一致性算法相比，Raft 使用一种更强的领导能力形式。比如：日志条目只从领导人发送给其的服务器。这种方式简化了对复制日志的管理，而且使得 Raft 算法更加容易理解。
- 领导选举：Raft 算法使用了一个随机定时器来选举领导人。这种方式只是在任何一致性算法都必须实现的心态机制上增加了一点机制。在解决冲突的时候会更加简单快捷。
- 集群成员变更：Raft 使用一种 joint consensus 方法来处理集群成员变更的问题，在这种方法下，处于调整过程中的两种不同的配置集群种大多数机器会有重叠，这使得集群在成员变更的时候依然可以继续工作。
## Replicated state Machines
共识算法是在复制状态机模型的背景下提出的。复制状态机可以简单分成两部分：状态机 + 复制。
- 状态机。可以理解成<font color="#ff0000">有状态的数据存储模块</font>，当给定输入，数据的状态会发生变化。最常见的状态机就是存储引擎，例如关系型数据库 mysql，给定增删改查的 sql 输入，执行后存储引擎状态会变；而 redis 等非关系型的数据库，client 发送的语句，redis 执行后状态也会发生变更。文件存储、对象存储也是类似。状态机可以抽象为存储数据的容器：写操作最终需要写到状态机，读操作需要从状态机获取数据。
- 复制。复制状态机模型是有多个状态机服务组成的一个分布式系统，通过一些技术手段实现集群种状态机的状态一致，彼此互为副本。同时可以容忍集群中部分机器宕机而不影响集群整体对外的服务。

复制状态机模型用在解决在分布式系统中强一致场景下的容错问题。(cp 系统)
如何实现集群中状态机的状态一致？通常基于复制日志实现，如下图所示。每个服务器都存储一系列指令的日志，并且按照日志的顺序执行。日志中的每条指令内容相同、顺序也相同，所以每个服务器执行的指令序列就是相同的，每一次执行操作都产生了相同的状态，最终保证了状态机一致。
![image.png|650x480](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118192647.png)
最典型的日志就是 mysql 的 binlog。对于没有操作日志的系统，可以将 client 发送的请求进行序列化为日志，在执行日志时，将内容反序列化即可。

> [!important]
> 共识算法就是要保证复制日志的一致性。

服务器上的共识模块接受客户端发送的请求，然后添加到自己的日志中，同时它和其他服务器上的共识模块进行通信来保证每一个服务器上的日志最终都以相同的顺序并包含相同的请求，即使有些服务器发生故障。
一旦指令日志被正确的复制，每个服务器的状态机按照日志顺序执行指令，然后输出结果被返回给客户端。
因此，整个服务器集群看起来就像是一个高可靠的单节点状态机。
共识算法的特性：
- 安全性保证（绝对不会返回一个错误的结果）：在非拜占庭错误情况下，包括网络延迟、分区、丢包、重复和乱序等错误都可以保证正确。
- 一定的可用性：集群中只要有大多数 (majority，超过一半) 的机器可运行并且能够相互通信、和客户端通信，就可以保证整体可用。因此，一个典型的包含 5 个节点的集群可以容忍 2 个节点的失败。
- 不依赖时序来保证一致性：物理时钟错误或者极端的消息延迟，最坏只会导致可用性问题，而不会产生一致性问题。
- 通常情况下，只要集群中的大多数 (majority) 节点响应一轮 rpc 就可以完成新日志的复制，小部分比较慢的节点不会影响系统整体的性能。
总结：共识算法保证各节点日志序列的强一致 -> 各节点顺序执行日志 -> 各节点状态一致 -> 集群整体对外强一致，并容忍少数节点宕机。
## What's wrong with Paxos?
两个明显的缺点：
1. 难以理解
2. 难以实现
## The Raft Consensus Algorithm
Raft 首先选举一个 leader，让 leader 负责管理复制日志。Leader 从 client 接收指令日志条目 (log entries)，把日志条目复制到其他服务器上，并告诉其他的服务器什么时候可以安全的将日志条目应用到他们自己的状态机中。
拥有一个 leader 可以大大简化对复制日志的管理。例如：leader 可以决定新的日志条目需要存放在日志序列的哪个位置而不需要和其他服务器商议，而且日志都只从 leader 流向其他服务器。
Leader 可以会发生故障，或者和其他服务器失去连接，这种情况下会选举一个新的 leader 出来。
通过引入 leader 的方式，Raft 将一致性问题分解成了三个相对独立的子问题，这些问题会在接下来的子章节中进行讨论：
1. Leader 选举：当现存的 leader 发生故障的时候，一个新的 leader 需要被选举出来。
2. 日志复制：leader 必须从客户端接收日志条目（log entries）然后复制到集群中的其他节点，并强制要求其他节点的日志和自己的保持一致。
3. 安全性：状态机安全性：如果一个服务器节点已经执行了 (apply) 一个确定的日志条目到它的状态机中，那么其他服务器节点不能在同一个日志索引位置执行一个不同的指令

### Raft Basic
一个 raft 集群包含若干个服务器节点；5 个服务器节点是一个经典的例子，这允许整个系统容忍 2 个节点失效。在任何时刻，每个服务器节点都处于这三个状态之一：领导者 (leader)、跟随者 (follower) 或者候选人 (condidate)。
在通常情况下，系统中只有一个领导人并且其他的节点全部都是跟随者。跟随者都是被动的：他们不会发送任何请求，只是简单的响应来自领导人或者候选人的请求。领导人处理所有的客户端请求（如果一个客户端和跟随者联系，那么跟随者会把请求重定向给领导人）。candidate 状态是用来选举新 leader 的。
节点状态与转移关系如图 figure 4. 所示。
![image.png|650x400](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118200007.png)

Raft 把时间分割成任意长度的任期 (term)，如 figure 5. 所示。
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118200435.png)

Term 用连续的整数标识。每一段任期从一次选举开始，一个或者多个候选人尝试竞争成为领导人。
如果一个候选人赢得候选，然后他就在接下来的任期内充当领导人的职责。在某些情况下，一次选举过程会造成选票的瓜分。在这种情况下，这一任期会以没有领导人结束；一次新的任期（和一次新的选举）很快就会重新开始。
<font color="#ff0000">Raft 保证了一个给定的任期内最多只有一个 leader。</font>

> [!warning]
> 注意不是一个给定的时刻最多只有一个 leader。因为可能存在网络分区，导致存在一个老 leader 和一个新选举出的 leader。

任期号 (term) 在 Raft 算法中充当逻辑时钟的作用，任期号使得服务器可以检测一些过期的信息：比如过期的领导人。
每个节点存储了一个当前任期号，任期号永远是单调递增的。每当服务器之间通过 rpc 发起通信的时候，都会交换当前任期号。如果一个服务器的当前任期号比其他人小，那么他会更新自己的任期号到较大的任期号值。如果一个候选人或者领导人发现自己的任期号过期了，那么他会立即恢复成跟随者的状态。如果一个节点收到一个包含过期任期号的请求，他会直接拒绝这个请求。
Raft 算法中通过 rpc 进行通信，并且实现基础的一致性算法只需要两种类型的 rpc：
1. 请求投票 RPC（Request Vote）: 由候选人 candidates 在候选期间发起。
2. 日志追加 RPC（AppendEntries）：由领导人 leader 发起，用来复制日志和提供心跳机制。

> 后续会为了在服务器之间传输快照，增加了第三种 RPC。

当服务器没有及时的收到 RPC 的响应时，会进行重试，并且他们能够并行的发起 RPC 来获得最佳的性能。
### Leader Election
Raft 使用一种心跳机制来触发领导人选举。当服务器程序启动的时候，初始化的都跟随者身份。
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118202608.png)

只要一个服务器节点可以从领导人或者候选人处收到有效的 Rpc，则会进行保持 follower 状态。领导人周期性的向所有跟随者发送心跳包（即不包含日志项内容的 AppendEntries RPC）来维持自己的 leader 权威。如果 follower 超过一段时间（election timeout）没有接收到任何消息，那么它就会认为系统中没有可用的领导人,就会尝试发起选举以选出新的领导人。
要开始一次选举过程，follower 首先增加自己的当前任期号 current term 并且转换到 candidates 状态。candidate 首先给自己投一票，然后他会并行地向集群中的其他服务器节点发送 RequestVote RPC 来给自己拉票。
candidates 会继续保持着当前状态直到以下三件事情之一发生：
1. 它自己赢得了这次选举
2. 其他人赢得了这次选举 (其他的节点成为了 leader)
3. 一段时间后没有人赢得这次选举
当 candidate 从整个集群的超过一半 (majority) 个数节点获得了选票，那么他就赢得了这次选举并成为领导人。每个投票者节点对一个任期号最多只投给一个 candidate，按照谁先来谁获胜的原则。majority 原则确保了一个任期最多只会有一个 candidate 赢得选举。一旦候选人赢得选举，他就立即成为领导人，然后他会向其他的服务器发送心跳消息来建立自己的权威并且阻止发起新的选举。
在等待投票结果的时候，候选人可能会从其他的服务器接收到 AppendEntries RPC。
1. 如果这个 rpc 里的的任期号>=候选人当前的任期号，那么候选人会承认这个 rpc 发起者的 leader 身份，将自己状态变为跟随者状态。
2. 如果这个 RPC 中的任期号比自己当前的任期号小，那么候选人就会拒绝这次的 RPC 并且继续保持候选人状态，继续选举流程。
第三种可能的结果是候选人既没有赢得选举也没有输：如果有多个跟随者同时成为候选人，那么选票可能会被瓜分以至于没有候选人可以赢得大多数人的支持（例如 5 节点集群中，4 个节点同时发起投票；或 3 个节点同时发起投票，瓜分成 2+2+1 的结果）。当这种情况发生的时候，每一个候选人都会超时，然后通过增加当前任期号来开始一轮新的选举。然而，如果没有其他机制的话，这种选票被瓜分的情况可能会一直持续下去。
Raft 算法使用随机选举超时时间的方法来确保很少会发生选票瓜分的情况，就算发生也能很快的解决。具体做法：<font color="#ff0000">每个节点的选举超时时间是一个固定时间区间里（例如 150-300 毫秒）的随机值</font>。这样可以把服务器都分散开以至于在大多数情况下只有一个服务器会第一个超时触发选举流程；然后他赢得选举并在其他服务器超时之前发送心跳包。
### Log Replication
一旦 leader 被选举出来，它就开始为客户端服务。客户端的每一个请求，都包含一条被复制状态执行的指令。
Leader 把这条指令作为一个新的日志条目追加到自己的日志序列中区，然后并行地发起 AppendEntries RPC 给其他的服务器，让他们复制这条日志条目。
当这条日志条目<font color="green">被安全的复制</font>，leader 就会应用 (apply) 这条日志条目到自己的状态机中，然后把执行的结果返回给 client。如果 followers 崩溃或者运行缓慢，又或者网络丢包，leader 会不停的重试 AppendEntries RPC (尽管已经回复了客户端)，直到所有的跟随者最终存储了所有的日志条目。
#### Logs Struct
日志以 figure 6. 的结构进行组织。
每条日志条目都存储了一条状态机指令 (类似 x<-5)，和任期号，该任期号是 leader 收到该指令时的任期号。<font color="#00b050">日志中的任期号是用来检查不同节点的日志序列是否出现了不一致的情况。</font>
每一个日志条目还有一个整型的 index，用来表明它在日志序列中的位置。
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118205644.png)

#### Logs replication
Raft 算法的日志复制流程中，日志的状态有两个重要概念：
1. <font color="#ff0000">Committed（已提交）</font>
2. <font color="#ff0000">Apply（应用）</font>
日志 Committed 表示了该日志已经被持久化，并且最终一定会被所有可用的状态机执行。简单来说，Committed 就表示该日志在集群内达成强一致、永不丢失了、或者说写入成功了。只有 Committed 的日志，状态机才能去执行 (apply) 它。
那问题来了，如何判断某一条日志是否可以被 Committed？
首先，这个判断工作是由 leader 来完成的。Leader 如何做判断？规则很简单：当 leader 将创建的日志条目复制到一半以上 (majority) 的服务器的时候，就可以断定该日志条目可以被 Committed 了（例如在 figure 6. 中条目 7）。
<font color="#00b050">同时，当一个日志被判断为可以 Committed，则 raft 同时保证该日志之前的所有日志也都可以被 Committed。</font>
因此 leader 只需要记录最后一个 Committed log 的 index。一旦 followers 知道一个日志条目已经被 Committed，那么他也会将这个日志条目应用到本地的状态机中（按照日志的顺序）。
Raft 算法在运行过程中保证了以下特性：
1. 在不同节点的日志序列中，如果两个日志条目拥有相同的索引和任期号，那么他们存储了相同的指令。
2. 在不同节点的日志序列中，如果两个日志条目拥有相同的索引和任期号，那么他们之前的所有日志条目也全部相同

> [!note]
> 这两个特性有点类似于 git，同一个项目代码的不同仓库、不同分支之间，如果某一个 commit id 相同，则之前的所有 commit id 都相同

Raft 是如何保证上述这两个特性的呢？
1. 第一个特性的保障手段：某一任期的 leader，在一个 index 位置最多创建一条日志，同时日志的位置永远不会改变 (index 不变)
2. 第二个特性的保障手段：AppendEntries RPC 做一致性检查：Leader 为每个 Follower 维护一个 next index，标示要给该 follower 发送的下一个 log 的 Index，初始化为 leader 最后一个 log index + 1。AppendEntries RPC 请求中会携带 2 个参数：[next index -1, next index-1 位置日志的 term]。Follower 接收到 AppendEntries RPC 之后会进行一些一致性检查，检查 AppendEntries 中指定的 LogIndex 和 term 与自己本地的日志是否一致，如果不一致就会向 Leader 返回失败。Leader 接收到失败之后，会将 nextId 减 1，重新进行发送，直到成功。这个回溯的过程实际上就是寻找 Follower 与 leader 日志的分叉点，然后 Leader 发送其后的 LogEntry。

> [!faq]
> 这里回答了前文：日志中的任期号是用来检查不同节点的日志序列是否出现了不一致的情况。

在集群正确运行过程中，领导人跟跟随者的日志保存着一致性，所以 AppendEntries RPC 的一致性检查从不失败。然而领导人崩溃的情况会让日志处于不一致的状态（老的 leader 可能还没有将所有日志条目复制给 follow）。这种不一致的问题会在 leader 和 follower 的一系列崩溃下加剧变的复杂。
Figure 7. 展示了 followers 的日志可能和新 leader 不同的各种情况。Follower 可能相比 leader 缺日志、多日志或者两种情况都有。丢失或者多出日志的情况可能会持续多个任期。
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118212108.png)

在 Raft 算法中，leader 是通过强制跟随者直接复制 leader 的日志来处理不一致问题的。这意味着在跟随者中的冲突的日志条目会被 leader 的日志覆盖。*Safety* 一节会阐述如何通过增加一些限制来使得这样的操作是安全的。
所以在下图所示的情况中，follower a 和 follower b 都会被覆盖，直到跟 leader 一致。
![image.png](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118211458.png)

通过上述日志复制机制，可以看出 raft 具有以下特性：
1. 只要集群中 majority 个节点可用，raft 就能正常工作
2. 在通常的情况下，新的日志条目可以在一次 RPC 中被复制给集群的 majority
3. 少数缓慢的 followers 不会影响整体的性能

## Safety
本节通过在 leader 选举时增加一些限制来完善 raft 算法。
### Election Restriction
raft 算法为了简化处理，对 leader 选举增加了限制条件，使得选举出来的新 leader 拥有所有之前任期中已经 Commited 的日志条目，而不需要 leader 从别的节点获取 Commited log。
即 raft 算法保证日志永远是单向流动的，即从 leader 到 follower，leader 永远不会覆盖写自己的日志，只会追加新日志。
限制内容如下：RequestVote RPC 中携带候选者的 log 信息，投票者需要根据此信息来判断候选者的日志是否比自己“全”，只有候选者的日志跟自己一样“全”或者比自己更“全”，才能投票给它，否则拒绝给该候选者投票。如果候选者得到了 majority 个数节点的认可，则说明候选者的日志比 majorty 节点中每个节点的日志都一样“全”或者更“全”，而这些 majority 节点中至少有一个节点拥有所有的 commited log（majority 总有交集），从而保证该候选者也一定拥有所有已经 commited log。
如何比较两个节点日志哪个更“全”？Raft 通过比较两份日志中最后一条日志条目的索引值 index 和任期号 term 定义谁的日志比较“全”。如果两份日志最后的条目的任期号 term 不同，那么任期号 term 大的日志更加“全”。如果两份日志最后的条目任期号相同，那么日志比较长的那个就更加“全”。
### Committing entries from previous terms
如上文介绍的那样，leader 判断<font color="#ff0000">当前任期内</font>的一条日志只要被存储到了 majority 个节点上，就可以被 Committed。
然而，这个判断规则并不适用于旧的 term 中的日志。即使旧 term 中的日志被复制到了 majority 个节点上，也依然有可能会被未来的 leader 给覆盖掉。
Figure 8. 解释了这个问题，为什么会被覆盖。
1. (a) 中，S1 被选举为 leader，并且它收到了一个 log entry，它只把这个 log entry 复制给一部分节点，它就崩溃了，这个 log entry 位于 term(2)，以及 S1 的 index 1 的位置。
2. (b) 中，S5 被选举为 leader（voted by S3, S4 and self）, 此时是 term (3), 它也收到了一个 log entry，同样位于 index 2 的位置。
3. (c) 中，S5 崩溃，S1 重启了，它又当选了 leader（term(4)），然后继续复制日志 term (2)，此时，term (2) 已经复制到了 majority 节点中，但是它并没有被 Committed。
4. (d) 中，S1 崩溃了，S5 重启，被重新当选为 leader（voted by S2, S3, S4）(term (5)), 此时它将 override 其他节点中 index 位置的 log entry。然而如果 S1 在 term (4) 中 Committed 了 term (2, 4) 中的 log entry，如 (e) 所示。此时 S5 就不能赢得选举。
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118213616.png)

为了消除图 8 里描述的情况，Raft 永远不会提交一个之前任期内的日志条目。
只有 leader 当前任期里的日志条目通过计算副本数目达到 majority 个断定可以被提交；一旦当前任期的日志条目以这种方式被提交，那么由于日志匹配特性，之前的老 term 的日志条目也都会被间接的提交。
## Follower and Condidate Crashes
跟随者和候选人崩溃后的处理方式比领导人要简单的多，并且他们的处理方式是相同的。
如果跟随者或者候选人崩溃了，那么后续发送给他们的 RPCs 都会失败。Raft 中处理这种失败就是简单地通过无限的重试；如果崩溃的机器重启了，那么这些 RPC 就会成功。
如果一个服务器在完成了一个 RPC，但是还没有响应的时候崩溃了，那么在他重新启动之后就会再次收到同样的请求。<font color="#ff0000">Raft 的 RPCs 都是幂等的</font>，所以这样重试不会造成任何问题。例如一个跟随者如果收到附加日志请求但是他已经包含了这一日志，那么他就会直接忽略这个新的请求。
## Timing and Availability
raft 算法中，时间不会影响正确性，但会影响可用性。
raft 算法中的 leader 选举对时间比较敏感。raft 要求时间满足以下关系：
broadcastTime ≪ electionTimeout ≪ MTBF

> [!info]
> broadcastTime：发送 rpc 并收到响应的平均时间
> electionTimeout：选举超时时间
> MTBF：平均故障间隔时间，就是对于一台服务器而言，两次故障之间的平均时间

其中 broadcastTime 和 MTBF 取决于系统环境，而 electionTimeout 是由我们配置的。
Raft 的 RPCs 需要接收方将一些信息持久化存储，所以 broadcastTime 大约是 0.5 毫秒到 20 毫秒，取决于存储设备和技术。
因此，electionTimeout 可能需要在 10 毫秒到 500 毫秒之间。
大多数的服务器的平均故障间隔时间 MTBF 都在几个月甚至更长，很容易满足时间的需求.
## Conclude
上述章节将基本的 raft 算法都描述完成了，论文总结了 raft 运行过程中的一些状态、以及 raft rpc 的内容、raft 中各节点需要遵守的规则:
![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118220106.png)

![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118220142.png)

![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118220202.png)

![image.png|650](https://raw.githubusercontent.com/four1er/tuchuan/main/img/20250118220233.png)

### 节点
对于节点而言，每个节点都应该保存以下数据：
- **currentTerm**：当前任期号
- **votedFor**：当前任期，该节点投票给了哪个节点
- **log[]**：日志数组，Leader 每次接收到客户端写请求时会生成一条日志数据并添加进日志数组中，日志中包含其在日志数组中**索引、当前任期和客户端指令**信息，Follower 接收 Leader 发来的同步日志数据并添加到日志数组的对应位置
- **commitIndex**：最后一条**被提交**的日志索引，Leader 将 log[] 中日志按顺序同步给 Follower 时，若日志被**过半节点**接收了则认为该日志已被提交，可更新 commitIndex 至最新被提交的日志索引
- **lastApplied**：最后一条**被应用**于节点状态机的日志索引，只有被提交的日志才能被应用于系统状态机，也就是说 log[] 中索引小于 commitIndex 的日志才能应用于状态机
- **nextIndex[]**：Leader 节点中存储需要给每个 Follower 同步的下一条日志索引，每次接收到 Follower 日志同步确认消息后将该 Follower 对应的 nextIndex 更新到下一条未被同步日志索引处，初始值为 Leader 的 log[] 长度 + 1
- **matchIndex[]**：Leader 节点中存储的每个 Follower 已经接收确认的最大日志索引，初始值为 0
### 日志
Raft 中每一个写操作被封装成一条日志 entry，**每一个 entry 由任期号、log 数组中索引、指令组成**。日志有三种状态：
- **未提交态**：1）Leader 节点接收到客户端发起的请求后将生成一条日志<term, index, cmd>，这条日志被添加进 log[] 中，此时这条日志处于未提交态 2) Leader 节点通过日志同步接口将未提交态的日志发送给 Follower，Follower 将日志添加进 log[] 中并向 Leader 节点发出确认响应，此时这条日志在 Follower 节点中也处于未提交态
- **提交态**：1）Leader 向 Follower 发起的日志同步消息得到了超过一半 Follower 的接收确认，Leader 将更新 commitIndex 至这个被确认的日志索引，**commitIndex 前的日志均为提交态** 2）Leader 更新 commitIndex 后向 Follower 同步 commitIndex，Follower 更新 commitIndex 与 Leader 一致，commitIndex 前的日志均为提交态
- **应用态**：日志被提交后就可以被应用于节点状态机进行指令执行，状态机按顺序应用被提交的日志，并更新 lastApplied，**lastApplied 之前的日志均为应用态**
# Raft 实现
https://www.youtube.com/watch?v=xztv-zIDLxc
Raft 将共识问题分解成三个相对独立的子问题：
1. Leader 选举
2. 日志复制
3. 安全性

所以我们将从以上三个方面梳理如何实现 raft 算法。
## Leader 选举实现
重新复习一下 leader 选举规则：
1. 在每一轮任期中，只会出现一个 leader，或者没有 leader。
2. 当 leader 下线后，集群开始进入新的任期。
3. 任何节点收到任期比自己任期大的请求时，需要马上跟随对方，并更新自己的任期。
4. 任何节点收到任期等于自己任期的数据追加请求的时候，需要马上跟随对方。
5. 在一轮任期的选举中，任何一个节点只能投给一个候选人。
6. 如果收到任期比自己小的请求直接丢失，否则必须回复。

选举过程：**候选人只会给自己投票，跟随者会一直投给第一个找他的候选人，只有得票超出一半的候选人才能成为领导者，所有人都必须跟随胜选的领导者。**
消息参数：
- **term:** candidate 发起选举时产生的新任期号，它的值为当前任期号 currentTerm + 1
- **candidateID**：候选人 ID
- **lastLogIndex**: candidate 节点 log[] 中最后一条日志的索引
- **lastLogTerm**：candidate 节点 log[] 中最后一条日志的任期号

消息响应：
- **term**: 接收到投票请求的节点（Voter) 会返回自己当前任期号，如果该值大于 candidate 的 currentTerm，candidate 需要将 currentTerm 更新到该值
- **voteGranted**：接收到投票请求的节点是否支持节点当选 Leader，true 表示支持，false 表示拒绝

实现逻辑：如果 term 小于 node 的 currentTerm => {currentTerm, false} 如果 term 大于 node 的 currentTerm => {currentTerm, true} 注意此时 voter 需要转变成 follower，并更新本地的 currentTerm 为最新值。如果 term 等于 node 的 currentTerm，这里需要判断一下，如果当前 voter 是第一次投票，或者 source 是之前投赞成票的对象，那么不介意再投一次赞成票。但是这里需要判断，message 中的 lastLogIndex 和 lastLogTerm 是否大于等于本地，是的情况下投出赞成票。代码实现如下：
```cpp
void TRaft::OnRequestVote(ITimeSource::Time now, TMessageHolder<TRequestVoteRequest> message) {
    if (message->Term < State->CurrentTerm) {
        auto reply = NewHoldedMessage(
            TMessageEx {.Src = Id, .Dst = message->Src, .Term = State->CurrentTerm},
            TRequestVoteResponse {.VoteGranted = false});
        Nodes[reply->Dst]->Send(std::move(reply));
    } else {
        if (message->Term > State->CurrentTerm) {
            State->CurrentTerm = message->Term;
            State->VotedFor = 0;
            StateName = EState::FOLLOWER;
            State->Commit();
        }

        bool accept = false;
        if (State->VotedFor == 0 || State->VotedFor == message->CandidateId) {
            if (message->LastLogTerm > State->LogTerm()) {
                accept = true;
            } else if (message->LastLogTerm == State->LogTerm() && message->LastLogIndex >= State->LastLogIndex) {
                accept = true;
            }
        }

        auto reply = NewHoldedMessage(
            TMessageEx {.Src = Id, .Dst = message->Src, .Term = State->CurrentTerm},
            TRequestVoteResponse {.VoteGranted = accept});

        if (accept) {
            VolatileState->ElectionDue = MakeElection(now);
            State->VotedFor = message->CandidateId;
            State->Commit();
        }

        Nodes[reply->Dst]->Send(std::move(reply));
    }
}
```
