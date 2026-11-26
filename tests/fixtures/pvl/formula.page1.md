subgraph $g_{1}$ into a super-vertex $s_{1}$. There are two parallel edges between $s_{1}$ and $v_{7}$ in Figure 5.

The graph shrink operator  $ \mathrm{GS}_{k_1}^{k_2}(\cdot) $ has several properties which will be useful for computing steiner connectivities. Firstly, applying the operator  $ \mathrm{GS}_{k_1}^{k_2}(\cdot) $ preserves the steiner connectivity for all edges in the resulting graph.

Property 1: Given a graph  $ G $ and two integers  $ k_1 \leq k_2 $, the steiner connectivity of each edge of  $ \text{GS}_{k_1}^{k_2}(G) $ when computed in  $ \text{GS}_{k_1}^{k_2}(G) $ is the same as that computed in  $ G $.

Secondly, the steiner connectivity for all edges of  $ \mathrm{GS}_{k}^{k}(G) $ is k. For example, all edges in Figure 5 have steiner connectivity 3.

Property 2: Given a graph  $ G $ and an integer  $ k $, every edge of  $ \mathrm{GS}_k^k(G) $ has steiner connectivity  $ k $.

Thirdly, multiple operations of  $ \mathrm{GS}_{k_1}^{k_2}(\cdot) $ can be chained together.

\textbf{Property 3:} Given a graph  $ G $ and four integers  $ k_1 \leq k_2 $ and  $ k_3 \leq k_4 $ such that  $ \max\{k_1, k_3\} \leq \min\{k_2, k_4\} $, we have  $ \mathrm{GS}_{k_3}^{k_4}(\mathrm{GS}_{k_1}^{k_2}(G)) = \mathrm{GS}_{\max\{k_1, k_3\}}^{\min\{k_2, k_4\}}(G) $.

Our Divide-and-Conquer Approach: ECo-DC. From Property 2, we know that the steiner connectivities of all edges of  $ \mathrm{GS}_{k}^{k}(G) $ are  $ k $. Moreover, from the definitions of steiner connectivity and the graph shrink operator, we know that all edges whose steiner connectivities are  $ k $ will be in  $ \mathrm{GS}_{k}^{k}(G) $. Thus, to compute steiner connectivities of all edges of  $ G $, it suffices to compute  $ \mathrm{GS}_{k}^{k}(G) $ for  $ k \in [1, \delta(G)] $. Instead of naively computing  $ \mathrm{GS}_{k}^{k}(G) $ independently for each  $ k \in [1, \delta(G)] $ which would take  $ \mathcal{O}\left(\delta(G) \times \mathsf{T}_{\mathrm{KECC}}(G)\right) $ time, we propose a divide-and-conquer approach based on the fact that  $ \mathrm{GS}_{k}^{k}(G) $ is entirely contained in  $ \mathrm{GS}_{k_{1}}^{k_{2}} $ if  $ k_{1} \leq k \leq k_{2} $.

Algorithm 3: ECo-DC(G)

1 Compute the degeneracy $\delta(G)$ of $G$;
2 Compute-DC$(G,1,\delta(G));$
3 Construct Hierarchy$(G,sc(\cdot,\cdot));$ /* See Algorithm 4 */;
4 return $\mathcal{T}$;

Procedure Compute-DC$(g,L,H)$

5 if $L=H$ then
6 $\left\lfloor\text{for each edge}(u,v)\in E(g)\text{do}sc(u,v)\leftarrow L;\right.$
7 else
8 $\left\{\begin{array}{l}\text{Choose an integer }M\text{ such that }L<M\leq H;\\\phi_{M}(g)\leftarrow\text{KECC}(g,M);\quad/*\text{Compute }M\text{-ECCs of }g\ast/;\\\text{Let }g_{1}\text{ be the graph obtained from }g\text{ by contracting each connected subgraph of }\phi_{M}(g)\text{ into a super-vertex, and }g_{2}\text{ be }\phi_{M}(g);\quad/* g_{1}=\text{GS}_{L}^{M-1}(G), g_{2}=\text{GS}_{M}^{H}(G)\ast/;\\\text{Compute-DC}(g_{1},L,M-1);\\\text{Compute-DC}(g_{2},M,H);\\\end{array}\right.$
9 $\phi_{M}(g)\leftarrow\text{KECC}(g,M);\quad/*\text{Compute }M\text{-ECCs of }g\ast/;\\\text{Let }g_{1}\text{ be the graph obtained from }g\text{ by contracting each connected subgraph of }\phi_{M}(g)\text{ into a super-vertex, and }g_{2}\text{ be }\phi_{M}(g);\quad/* g_{1}=\text{GS}_{L}^{M-1}(G), g_{2}=\text{GS}_{M}^{H}(G)\ast/;\\\text{Compute-DC}(g_{1},L,M-1);\\\text{Compute-DC}(g_{2},M,H);\\\end{array}\right.$

The pseudocode of our approach is shown in Algorithm 3, denoted ECo-DC. It first computes the degeneracy  $ \delta(G) $ of  $ G $ (Line 1), and then invokes procedure Compute-DC with input  $ (G, 1, \delta(G)) $ to compute the steiner connectivities of all edges (Line 2), while Line 3 constructs the hierarchy tree and will be discussed in Section 4.2. The input to Compute-DC consists of a graph  $ g $ and an interval  $ [L, H] $. If  $ L = H $, then the steiner connectivities of all edges of  $ g $ are set as  $ L $ (Lines 5–6). Otherwise, an integer  $ M $ is chosen such that  $ L < M \leq H $ (Line 8), then the set  $ \phi_M(g) $ of  $ M $-ECCs of  $ g $ is computed (Line 9) and two graphs  $ g_1 $ and  $ g_2 $ are obtained from  $ g $ based on  $ \phi_M(g) $ (Line 10), and finally the algorithm continues on  $ g_1 $ (Line 11) and on  $ g_2 $ (Line 12).



We prove by the following lemma that when initially invoking Compute-DC with graph G and interval  $ [1,\delta(G)] $, the graph g being processed for each recursion with interval  $ [L,H] $ is  $ \mathrm{GS}_{L}^{H}(G) $.

Lemma 4.1: For Compute-DC, if the input graph $g$ is $\mathrm{GS}_{L}^{H}(G)$, then the two graphs $g_{1}$ and $g_{2}$ obtained at Line 10 are exactly $\mathrm{GS}_{L}^{M-1}(G)$ and $\mathrm{GS}_{M}^{H}(G)$, respectively.

![](imgs/img_in_image_box_635_522_1116_729.jpg)

Figure 6: Recursion tree

Based on Lemma 4.1, the recursion tree of invoking Compute-DC with input  $ (G, 1, \delta(G)) $ is as shown in Figure 6.

The correctness and time complexity of Algorithm 3 are proved by the two theorems below.

Theorem 4.1: Algorithm 3 correctly computes the steiner connectivity for all edges of G.

Theorem 4.2: The time complexity of Algorithm 3 is  $ O(h \times \text{TKECC}(G)) $, where h is the height of the recursion tree in Figure 6.

Near-Optimal Time Complexity. Algorithm 3 correctly computes the steiner connectivities of all edges regardless of the choice of  $ M $ at Line 8, as long as  $ L < M \leq H $. Yet, the time complexity of Algorithm 3 would vary for different choices of  $ M $. For example, if  $ M $ is always set as  $ L + 1 $ or always set as  $ H $, then the height of the recursion tree would be  $ \delta(G) $ and thus the time complexity of Algorithm 3 would be  $ O(\delta(G) \times \text{T}_{\text{KECC}}(G)) $ on the basis of Theorem 4.2. To make the time complexity as low as possible, we will need to reduce the height of the recursion tree. Thus, we propose to set  $ M $ as  $ \left\lfloor \frac{L + H}{2} \right\rfloor $, and prove in the following theorem that the time complexity of Algorithm 3 then becomes  $ O\left( (\log \delta(G)) \times \text{T}_{\text{KECC}}(G) \right) $.

Theorem 4.3: By setting  $ M = \left\lceil \frac{L+H}{2} \right\rceil $, the time complexity of Algorithm 3 is  $ O\left( (\log \delta(G)) \times \text{TK}_{\text{ECC}}(G) \right) $.

Following the above theorem, we set  $ M = \left\lceil \frac{L + H}{2} \right\rceil $ in Algorithm 3. The time complexity of ECo-DC, which is  $ O\left( (\log \delta(G)) \times T_{KECC}(G) \right) $, is optimal up to a logarithmic factor  $ \log \delta(G) $. This is because the time complexity of ECo-decomposition cannot be lower than  $ T_{KECC}(G) $, as ECo-decomposition also implicitly computes the k-ECCs of G; specifically, the k-ECCs of G can be obtained from the hierarchy tree in time linear to the sizes of the k-ECCs.