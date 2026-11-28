![](imgs/img_in_image_box_103_165_584_795.jpg)

Figure 7: Running example of ECo-DC

Example 4.1: Here, we apply ECo-DC on the graph G in Figure 2 as an example. Figure 7 indicates the whole running process of ECo-DC on G, where the top-most part is G itself. The degeneracy is  $ \delta(G) = 4 $. Then, we compute the steiner connectivities of all edges of G by invoking Compute-DC with input G and  $ [L, H] = [1, 4] $. Here,  $ GS_1^4(G) $ is the same as G. As  $ L \neq H $ and  $ \left\lceil \frac{L + H}{2} \right\rceil = 3 $, we compute the 3-ECCs of G and obtain the subgraphs induced by  $ S_1 = \{v_1, v_2, \ldots, v_9\} $ and  $ S_2 = \{v_10, \ldots, v_{13}\} $, respectively. Thus, we obtain the two graphs  $ GS_1^2(G) $ and  $ GS_3^4(G) $ as shown in the middle layer of Figure 7. The computation continues on these two graphs with intervals  $ [1, 2] $ and  $ [3, 4] $, respectively.

The graph  $ \mathrm{GS}_{3}^{4}(G) $ is composed of the two 3-ECCs of G as shown in right part of the middle layer of Figure 7. We compute the 4-ECCs of  $ \mathrm{GS}_{3}^{4}(G) $, and obtain the subgraph induced by vertices  $ \{v_{1}, v_{2}, \ldots, v_{5}\} $. Thus, all edges among vertices  $ \{v_{1}, v_{2}, \ldots, v_{5}\} $ have steiner connectivities 4 as indicated in  $ \mathrm{GS}_{4}^{4}(G) $, while the other edges have steiner connectivities 3 as demonstrated in  $ \mathrm{GS}_{3}^{3}(G) $.

The graph  $ \mathrm{GS}_{1}^{2}(G) $ is obtained by contracting each of  $ S_{1} $ and  $ S_{2} $ into a super-vertex as shown in the left part of the middle layer of Figure 7. In  $ \mathrm{GS}_{1}^{2}(G) $, there are two parallel edges between  $ s_{1} $ and  $ s_{2} $, corresponding to edges  $ (v_{9}, v_{11}) $ and  $ (v_{5}, v_{12}) $, respectively. As  $ \mathrm{GS}_{1}^{2}(G) $ is 2-edge connected, the steiner-connectivities of  $ (v_{9}, v_{11}) $ and  $ (v_{5}, v_{12}) $ are 2.

### 4.2 Constructing the Hierarchy Tree

Given the steiner connectivities of all edges of a graph $G$, Algorithm 4 constructs the hierarchy tree of ECo-decomposition of $G$ in a bottom-up manner. The main idea is as follows. First, the hierarchy tree $\mathcal{T}$ is initialized as a forest of singleton nodes. Then, for each edge $(u, v) \in E(G)$ in non-increasing order regarding $sc(\cdot, \cdot)$, we identify the tree in $\mathcal{T}$ (specifically, the root $r_u$ of the tree) containing $u$ and the tree (specifically, the root $r_v$ of the tree) containing $v$. If $u$ and $v$ are already in the same tree $(i.e., r_u = r_v)$, then we do nothing. Otherwise, we merge the two trees into one in $\mathcal{T}$, with the root of this newly formed tree having weight $sc(u, v)$.

Algorithm 4: ConstructHierarchy

Input: A graph G with  $ sc(u, v) $ for each edge  $ (u, v) $
Output: The hierarchy tree of ECo-decomposition of G

1 Initialize an empty hierarchy tree  $ \mathcal{T} $;
2 for each vertex  $ u \in V(G) $ do Insert a singleton node u into  $ \mathcal{T} $;
3 for each edge  $ (u, v) \in E(G) $ in non-increasing  $ sc(u, v) $ order do
4 Let  $ r_u $ (resp  $ r_v $) be the root of the tree in  $ \mathcal{T} $ containing u (resp v);
5 if  $ r_u = r_v $ then continue;
6 else if both  $ r_u $ and  $ r_v $ are ECC nodes with weight  $ sc(u, v) $ then
7  $ \left\lfloor \text{Merge } r_u \text{ and } r_v \text{ into a single ECC node};\right. $
8 else if none of  $ r_u $ or  $ r_v $ is an ECC node with weight  $ sc(u, v) $ then
9  $ \left\lfloor \text{Create a new ECC node in } \mathcal{T}\text{ with weight }sc(u, v), \text{ and add } r_u \text{ and } r_v \text{ as its children};\right. $
10 else
11 Without loss of generality, assume  $ r_u $ is an ECC node with weight  $ sc(u, v) $, and add  $ r_v $ as a child of  $ r_u $ in  $ \mathcal{T} $;

The pseudocode of constructing the hierarchy tree is illustrated in Algorithm 4, denoted by ConstructHierarchy. The input of the algorithm is a graph  $ G $ with  $ sc(u, v) $ precomputed for each edge  $ (u, v) $. It first initializes an empty hierarchy tree (Line 1), and creates a single-node tree in  $ \mathcal{T} $ for each vertex of  $ G $ (Line 2). Then, the trees in  $ \mathcal{T} $ will be merged with each other to form ECC nodes in the hierarchy tree. For each edge  $ (u, v) \in E(G) $ sorted by  $ sc(u, v) $ in non-increasing order (Line 3), the roots of the trees in  $ \mathcal{T} $ containing node  $ u $ and node  $ v $ are found, represented by  $ r_u $ and  $ r_v $ respectively (Line 4). If  $ r_u = r_v $, it implies that vertices  $ u $ and  $ v $ have already been merged into the same tree so that the algorithm skips the current edge (Line 5); otherwise, the algorithm merges  $ r_u $ and  $ r_v $ into a single tree based on the following three cases. (1) If both  $ r_u $ and  $ r_v $ are ECC nodes with weight  $ sc(u, v) $, it merges  $ r_u $ and  $ r_v $ into a single ECC node (Lines 6–7). (2) If neither  $ r_u $ nor  $ r_v $ is an ECC node with weight  $ sc(u, v) $, it creates a new ECC node in  $ \mathcal{T} $ with weight  $ sc(u, v) $ whose children are  $ r_u $ and  $ r_v $ (Lines 8–9). (3) The last situation is that one of  $ r_u $ or  $ r_v $ is an ECC node with weight  $ sc(u, v) $ and the other is not; note that, if the other one is an ECC node, then its weight must be larger than  $ sc(u, v) $. Assume that  $ r_u $ is the one with weight  $ sc(u, v) $,  $ r_v $ would be added as a child of  $ r_u $. Similar steps would be applied to the situation where  $ r_v $ is the one with weight  $ sc(u, v) $ (Lines 10–11).

The most time-consuming operation in Algorithm 4 is Line 4, which aims to find the root of the tree that contains a node u.