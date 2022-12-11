# Algorithm_Notes
## 1.1 Backtracking 
### What is backtracking?
Backtracking is Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally,  
one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point in time.  

Therefore, Backtracking is mainly used to slove: 1) combination. 2) permutation. 3) partitioning(partitioning substring or subset).  
Examples: Leetcode 77 (combinations). Leetcode 39 (combination sum).Leetcode 17 (Letter Combinations of a Phone Number). Leetcode 131 (Partitioning Panlindrome).
Leetcode 332 (Reconstruct Itinerary)

`1. Find stopping Criteria. 2. Understand single-iteration backtrack logic.`<br />
Sample Code for leetcode 17: <br />
<img width="260" alt="17_phone_number_combination" src="https://user-images.githubusercontent.com/74223059/206390531-ac2a9800-012c-424e-8527-e15c266dfbaf.png">
<br />
Note 1: if use `if else` logic for the stopping, No need to use return.  

Note 2：for backtracing, if use `back(start+1,com+l)`, no need to reset the current combination, which is `com=com[:-1]`.

Sample Code for Leetcode 131 (Partitioning Panlindrome) (Where the single-iteration becomes a bit more complicated):

<img src='https://user-images.githubusercontent.com/74223059/206667833-903f54e3-16c5-4215-ac6a-436c30e62f5f.png' width="400" height='300'/> <img src='https://user-images.githubusercontent.com/74223059/206667782-f95d4a97-27e9-4cc9-ab20-bb155bfdd1ed.png' width="400" height='300'/> 

## 1.2 Dynamic Programming 
The optimization of dynamic programming is to record last-step situation (every step, every possibility) so to avoid repeative caculation.
### Steps: <br />
1. Define dp table meaning. 
2. Find the transfer function.(How dp table is updated from t to t+1)
3. Think through the traversing and updating process.

Note 1 : Care Initializaton and Boundary (list index out of range) .
Note 2 : For some problems (searching matrix), Dynamic Programming is an optimized Algorithm for DFS with memorization. Using DFS may cause time limt to exceed.

Classic dynamic programming problem:
1) Least Cost Route/Arrangement.(DFS may be applicable in some cases).eg. Fibonacci numbers (509), Climbing stairs (70,746), Unique Paths (62,63)  
2) best time to sell stocks (121,122,123,188,309,714)/ largest temperature difference.
3) Packaging problems.(1049)
4) Subset Problem (674，516)






