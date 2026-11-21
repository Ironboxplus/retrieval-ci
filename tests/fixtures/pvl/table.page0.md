Table 1: Input tables for our running example.

(a) Person table.

(b) Follows table.



<table><tr><td colspan="2">Person</td><td colspan="3">Follows</td></tr><tr><td>ID</td><td>name</td><td>ID1</td><td>ID2</td><td>year</td></tr><tr><td>101</td><td>Mahinda</td><td>101</td><td>202</td><td>2021</td></tr><tr><td>202</td><td>Karim</td><td>303</td><td>404</td><td>2019</td></tr><tr><td>303</td><td>Carmen</td><td>101</td><td>303</td><td>2021</td></tr><tr><td>404</td><td>Zhang</td><td>202</td><td>303</td><td>2020</td></tr><tr><td></td><td></td><td>101</td><td>404</td><td>2021</td></tr></table>

Table 2: Extended tables. RID columns are abbreviated as VR and are in gray to indicate that they are not materialized. The RID( $ ID_{i} $) columns of Follows' are abbreviated as Ri.

(a) Extended Person table. (b) Extended Follows table.



<table><tr><td colspan="3">Person</td><td colspan="6">Follows&#x27;</td></tr><tr><td>VR</td><td>ID</td><td>name</td><td>VR</td><td>R1</td><td>ID1</td><td>R2</td><td>ID2</td><td>year</td></tr><tr><td>0</td><td>101</td><td>Mahinda</td><td>0</td><td>0</td><td>101</td><td>1</td><td>202</td><td>2021</td></tr><tr><td>1</td><td>202</td><td>Karim</td><td>1</td><td>2</td><td>303</td><td>3</td><td>404</td><td>2019</td></tr><tr><td>2</td><td>303</td><td>Carmen</td><td>2</td><td>0</td><td>101</td><td>2</td><td>303</td><td>2021</td></tr><tr><td>3</td><td>404</td><td>Zhang</td><td>3</td><td>1</td><td>202</td><td>2</td><td>303</td><td>2020</td></tr><tr><td></td><td></td><td></td><td>4</td><td>0</td><td>101</td><td>3</td><td>404</td><td>2021</td></tr></table>

of the row  $ r_p \in P $ to which  $ r_f $ has the foreign key. This column is visible only to the system and not to users. RIDs in columnar RDBMSs serve as system-level pointers and can be used to compute the locations of rows in storage. Therefore the RID $ (A_{ft_1}, ..., A_{ft_t}) $ column stores for each  $ r_f $ the pointer to the matching  $ r_p $. Multiple joins on  $ F $ can be predefined, which is common for relationship tables that represent many-to-many joins between two entity tables.

EXAMPLE 1. Table 1 shows a simple database with two tables, a Person(ID, name) table and a Follows(ID1, ID2, year) table, that will serve as our running example. The ID1 and ID2 columns in Follows are both foreign keys to the ID column of Person. Table 2 shows the extended Follows table (as Follows') when a user predefines the Person. ID = Follows. ID1 and Person. ID = Follows. ID2 joins. The Follows table is extended with RID(ID1) and RID(ID2) columns (abbreviated as RID1 and RID2) that contain the RIDs of the rows in Person that match the values in the ID1 and ID2 columns, respectively. Both Person and Follows tables also have RID columns (abbreviated as VR) that show the contiguous RIDs of the rows in these tables. These are shown in gray to indicate that unlike RID(ID1) and RID(ID2) columns, they are not materialized in storage.

## 5 SJOIN: THE SIP OF RIDS

Our implementation of predefined joins consists of two steps:

Step 1: Rule-based query optimization. Algorithm 1 gives the pseudocode of our first rule-based optimization algorithm (a second one is described in Section 6.2). We recursively traverse the system's default logical plan T for a query (lines 1 and 2) and find each join operator that evaluates a predefined join from F to P. In our implementation, these are HashJoin operators because DuckDB evaluates equality joins with HashJoin. Upon finding these HashJoins, we perform one of two actions. We review the first case here and will review the second case in Section 6.1:

Join Replacement Case 1 (Lines 8-10): F is on the build and P is on the probe side. In this case, we make the following changes to T:

• HashJoin is replaced with a new operator called SJoin (line 9).

Algorithm 1 ReplaceHashJoinsWithSJoins.T.op refers to the root operator in a (sub-) plan tree T.

input: DuckDB's optimized plan tree T
output: A plan tree T* possibly with SJoin/SJoinIdxR operators.
1: T* ← ReplaceHashJoinsWithSJoins(T);
2: T* ← ReplaceHashJoinsWithSJoins(T);
3: if ((F, P) = FindPredefinedJoin(T.op)) ≠ NULL then
4:     return RewriteJoinAndScans(T, F, P);
5: return T;
6:
7: function RewriteJoinAndScans(T, F, P):
8: if (F is on the build and P is on the probe side) then
9:     (i) replace T.op with SJoin;
10:     (ii) replace the scan of P with ScanSJ of P.
11: else if P is on the build and F is on the probe side and the appropriate RID index on F exists then
12:     (i) replace T.op with SJoinIdxR;
13:     (ii) replace the scan of F with ScanSJ of F.

- Scan(F) operator (on the build side sub-tree) is modified to scan the materialized RID column of F (omitted in Algorithm 1).

• Every probe-side Scan(P) operator is replaced with a modified scan operator, which we call ScanSJ (for semi join) (line 10).

Step 2: Sideways information passing during query evaluation: Suppose an SJoin operator has replaced a HashJoin operator in T and a ScanSJ(P) has replaced the standard Scan(P) in the Join Replacement Case 1. Then during query evaluation we use sip to pass information from SJoin to ScanSJ(P) as follows. SJoin is a specialized hash join operator that first reads all of the tuples from its build side. These tuples contain materialized RID values in F and point to the tuples in P. SJoin then constructs two bitmask filters:

- Zone bitmask: For each zone of P, i.e., a block of tuples on disk, indicates whether the zone has any matching tuples joining with F. This bitmask contains 1 bit for each zone and is constructed by taking the modulo of the RIDs with the zone size.

- Row bitmask: Indicates whether each row  $ r_p $ of P joins with an F tuple. This bitmask contains  $ |P| $ many bits and is constructed by directly setting the positions of the seen RIDs to 1.

SJoin then passes both of these bitmasks to all of the ScanSJ(P) operators in its probe side recursively. The zone bitmask is used to skip over scanning zones of P whose bits are 0. For zones with matching tuples, The ScanSJ operator scans the zone into vectors as regular scan operator and adds a new RID vector to the intermediate tuples that store the RIDs of the scanned tuples. Finally, ScanSJ attaches the row bitmask of this zone as a selector vector, which filters out the P tuples without matching F tuples, to the intermediate tuples.

Note that passing bitmask filters sideways from SJoin operator down to all ScanSJ(P) operators is a safe operation in our setting of select-project-join queries. Observe that the bitmask filter generated by SJoin is the set of RIDs of P that will successfully join with F tuples in the SJoin. Let P.R be the (virtual) RID column of P. One can therefore think of the bitmask filter as a runtime predicate on P.R. One can further think of applying this predicate as the last operator of the probe-side of SJoin. Therefore by passing the bitmask