---
{"title":"ZooKeeper","auther":"four1er","created_at":"2025-01-23 10:46","last modify":"2025-01-23 10:46","file path":"藏经阁/Distributed System/6.824/ZooKeeper.md","tags":["distributed_sytem","zookeeper"],"dg-publish":true,"permalink":"/藏经阁/Distributed System/6.824/ZooKeeper/","dgPassFrontmatter":true,"created":"2025-02-05T10:35:59.677+08:00","updated":"2025-02-10T21:27:35.759+08:00"}
---

# 线性一致
如果一个服务是线性一致的，那么它表现的就像只有一个服务器。一个线性一致系统中的执行历史中的操作是非并发的。

# 为什么要有 Zookeeper
这个标题还有另一个表达方式，Zookeeper 解决了什么问题？
在回答这个问题之前，我们先了解一下分布式系统的一个痛点问题。作为一个分布式系统，我们需要具有一定的容错处理，这意味着我们通常需要通过多副本来完成容错，所以一个 Zookeeper 可能会有 3 个、5 个或者 7 个服务器，而这些服务器是需要花钱的，很明显，一个 7 个节点的服务器集群是要比 1 个节点的 " 集群 " 贵 7 倍。
所以这里有个问题，如果花了 7 个服务器来运行多副本服务，那么是否能通过这 7 台服务器得到 7 倍的性能？

> [!attention]
> 在接下来的讨论中，我们将会把 Zookeeper 看成一个类似于 Raft 的多副本系统。Zookeeper 实际上运行在 Zab 之上，从我们的角度来看，Zab 几乎与 Raft 是一样的。并且这里我们只看多副本系统的性能，并不关心 Zookeeper 的具体功能。
