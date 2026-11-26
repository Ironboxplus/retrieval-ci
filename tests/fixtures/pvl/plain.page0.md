# Hybrid Blockchain Database Systems: Design and Performance

Zerui Ge

School of Computing

National University of Singapore

gezerui@comp.nus.edu.sg

Dumitrel Loghin

School of Computing

National University of Singapore

dumitrel@comp.nus.edu.sg

Beng Chin Ooi

School of Computing

National University of Singapore

ooibc@comp.nus.edu.sg





Pingcheng Ruan

School of Computing

National University of Singapore

ruanpc@comp.nus.edu.sg

## ABSTRACT

With the emergence of hybrid blockchain database systems, we aim to provide an in-depth analysis of the performance and trade-offs among a few representative systems. To achieve this goal, we implement Veritas and BlockchainDB from scratch. For Veritas, we provide two flavors to target the crash fault-tolerant (CFT) and Byzantine fault-tolerant (BFT) application scenarios. Specifically, we implement Veritas with Apache Kafka to target CFT application scenarios, and Veritas with Tendermint to target BFT application scenarios. We compare these three systems with the existing open-source implementation of BigchainDB. BigchainDB uses Tendermint for consensus and provides two flavors: a default implementation with blockchain pipelining and an optimized version that includes blockchain pipelining and parallel transaction validation. Our experimental analysis confirms that CFT designs, which are typically used by distributed databases, exhibit much higher performance than BFT designs, which are specific to blockchains. On the other hand, our extensive analysis highlights the variety of design choices faced by the developers and sheds some light on the trade-offs that need to be done when designing a hybrid blockchain database system.

PVLDB Reference Format:

Zerui Ge, Dumitrel Loghin, Beng Chin Ooi, Pingcheng Ruan, and Tianwen Wang. Hybrid Blockchain Database Systems: Design and Performance. PVLDB, 15(5): 1092 - 1104, 2022.

doi:10.14778/3510397.3510406

## PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at https://github.com/nusdbsystem/Hybrid-Blockchain-Database-Systems.

## 1 INTRODUCTION

In the last few years, a handful of systems that integrate both distributed databases and blockchain properties have emerged in the academic database community [16, 19, 20, 31, 33, 38]. These systems,

Tianwen Wang

School of Computing

National University of Singapore

tianwenw@nus.edu.sg



termed hybrid blockchain database systems, are either adding database features to existing blockchains to improve performance and usability or implement blockchain features in distributed databases to improve their security [36]. However, there is little comparison among these systems. In this paper, we aim to fill this gap by providing an in-depth analysis of a few representative hybrid blockchain database systems.

To achieve this goal, we first have to overcome a challenge: only one such system, namely BigchainDB [20] is open-source. Moreover, by the time of writing this paper, we did not manage to get the source code of any other hybrid blockchain database system. Hence, we undertake the tedious task of implementing Veritas [19] and BlockchainDB [16]. By doing so, we gain the flexibility of changing some parts of the systems to better compare them to other systems. For example, we can change the underlying database from a relational SQL type to a NoSQL type. Or we can change the consensus mechanism from crash fault-tolerant (CFT) to Byzantine fault-tolerant (BFT).

The original design of Veritas uses Apache Kafka [17], which is a CFT service, as the broadcasting service among server nodes. That is, when a server node needs to update its local shared database as a result of a transaction's execution, it sends this update to all the other server nodes via the Kafka service. Moreover, the other server nodes send their agreement or disagreement regarding that update via the Kafka service. Apache Kafka uses a primary backup mechanism to achieve CFT. Hence, it has high efficiency at the cost of decreased liveness. On the other hand, most blockchain systems adopt a BFT consensus mechanism. Hence, we also implement a version of Veritas where Tendermint [3], which is a BFT consensus, is used as middleware, similar to BigchainDB [20].

BlockchainDB [16] implements a shared (and sharded) database on top of a classic blockchain. The database interface provides a simple key-value store API to the user, similar to Veritas and BigchainDB. The blockchain layer is flexible, being able to interact with different blockchains, such as Ethereum [4] and Hyperledger Fabric [18]. In this paper, as well as in [16], BlockchainDB uses Ethereum as its underlying blockchain.

BigchainDB introduces two optimizations. The first optimization, called blockchain pipelining, allows nodes to vote for a new block while the current block of the ledger is still undecided. Each node will just reference its last decided block when voting for a new block. By doing so, the system avoids waiting for blocks to be fully committed before proposing new blocks. Second, BigchainDB