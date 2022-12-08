# Algorithm_Notes
## 1.1 Backtracking 
### What is backtracking?
Backtracking is Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally,  
one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point in time.  

Therefore, Backtracking is mainly used to slove: 1) combination. 2) permutation. 3) partitioning(partitioning substring or subset).  
Examples: Leetcode 77 (combinations). Leetcode 39 (combination sum).Leetcode 17 (Letter Combinations of a Phone Number)  

`1. Find stopping Criteria. 2. Understand single-iteration backtrack logic.`<br />
Sample Code for leetcode 17: <br />
<img width="260" alt="17_phone_number_combination" src="https://user-images.githubusercontent.com/74223059/206390531-ac2a9800-012c-424e-8527-e15c266dfbaf.png">

Note 1: if use `if else` logic for the stopping, No need to use return.  

Note 2：for backtracing, if use `back(start+1,com+l)`, no need to reset the current combination, which is `com=com[:-1]`.






