# Algorithm_Notes
## 1.1 Backtracking 
### What is backtracking?
Backtracking is Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally,  
one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point in time.  

`1. Find stopping Criteria. 2. Understand single-iteration backtrack logic.`<br />

How Backtracking Works:

Choose: Make a choice that helps progress towards the solution.

Constrain: Check if the choice made in step 1 is valid. If valid, continue; else undo the choice and go back to step 1.

Goal: We have a valid solution once we have exhausted all choices.

Let's break it down:

1. Choose:

We need to make a choice at each step of reaching our goal. This choice will help us progress towards our solution.

2. Constrain:

After making a choice, we need to check whether the choice led us to a valid solution or not. These checks are often problem-specific.

3. Goal:

This is the final step where we have made all the choices and checked them. If the goal is reached, we have a valid solution. Otherwise, we backtrack.

Backtracking problems can often be solved with recursion or a stack since both of these can be used to implement depth-first search.


Sample Code for leetcode 17: <br />
<img width="260" alt="17_phone_number_combination" src="https://user-images.githubusercontent.com/74223059/206390531-ac2a9800-012c-424e-8527-e15c266dfbaf.png">
<br />
Note 1: if use `if else` logic for the stopping, No need to use return.  

Note 2：for backtracing, if use `back(start+1,com+l)`, no need to reset the current combination, which is `com=com[:-1]`.

Sample Code for Leetcode 131 (Partitioning Panlindrome) (Where the single-iteration becomes a bit more complicated):

<img src='https://user-images.githubusercontent.com/74223059/206667833-903f54e3-16c5-4215-ac6a-436c30e62f5f.png' width="400" height='300'/> <img src='https://user-images.githubusercontent.com/74223059/206667782-f95d4a97-27e9-4cc9-ab20-bb155bfdd1ed.png' width="400" height='300'/> 

Identifying Backtracking Problems:

The problem seems to ask for all valid configurations of a particular arrangement (like permutations, combinations, subsets).
The problem uses language like "all valid placements" or "all possible configurations."
Any problem that requires visiting all solutions to subproblems and combines their results
Examples of Backtracking Problems:

Permutations (LeetCode #46): This problem asks for all permutations of a distinct array of numbers. You can solve this by swapping elements to generate permutations and backtracking to undo the swaps to generate new permutations.

Subsets (LeetCode #78): This problem requires you to find all possible subsets of a distinct array of numbers. You can solve this by recursively adding elements to a subset, adding it to your solution set, and then backtracking by removing the element and attempting to add the next one.

N-Queens (LeetCode #51): The N-Queens puzzle is the problem of placing N queens on an N×N chessboard such that no two queens threaten each other. This is a classic backtracking problem, where you place queens row by row, and if you find an invalid placement, you backtrack and move the previous queen.
  
Examples: Leetcode 77 (combinations). Leetcode 39 (combination sum).Leetcode 17 (Letter Combinations of a Phone Number). Leetcode 131 (Partitioning Panlindrome).
Leetcode 332 (Reconstruct Itinerary)
## 1.2 Dynamic Programming 
**Dynamic Programming Tutorial**

Dynamic Programming (DP) is an algorithmic technique used to solve an optimization problem by breaking it down into simpler subproblems and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems. It is both a mathematical optimization method and a computer programming method. 

**Understanding Dynamic Programming:**

There are two key attributes that a problem must have in order for dynamic programming to be applicable: optimal substructure and overlapping subproblems.

1. **Optimal Substructure:** This means an optimal solution can be constructed efficiently from optimal solutions of its subproblems.

2. **Overlapping Subproblems:** This means a recursive solution contains a small number of distinct subproblems that are repeated many times.

Dynamic Programming typically applies to optimization problems, like shortest path, longest increasing subsequence, or knapsack problems.

**Steps for Solving DP Problems:**

1. **Characterize the Structure of an Optimal Solution**
   
2. **Define the Value of an Optimal Solution Recursively in Terms of Smaller Values**

3. **Compute the Value of an Optimal Solution (typically in a bottom-up fashion)**

4. **Construct an Optimal Solution to the Problem from the Computed Information**

**Examples and How to Solve Them:**

**Example 1: Fibonacci Sequence**

The Fibonacci sequence is a classic example where Dynamic Programming can be used. With a naive recursive algorithm, there are many overlapping subproblems. By using Dynamic Programming, we can avoid solving these overlapping subproblems repeatedly by simply storing their results in an array.

**Example 2: 0/1 Knapsack Problem**

Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. This problem can be solved by constructing a DP table that contains maximum values of knapsack for all capacities from 0 to W and for all items. 

**Example 3: Longest Common Subsequence (LeetCode #1143)**

Given two strings text1 and text2, return the length of their longest common subsequence. A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters. This problem can be solved by using Dynamic Programming to find LCS lengths for all combinations of first i characters of text1 and first j characters of text2. 

Dynamic Programming is a powerful technique that allows one to solve different types of problems in time O(n^2) or O(n^3) for which a naive approach would take exponential time. The idea is to simply store the results of subproblems, so that we do not have to re-compute them when needed later. This simple optimization reduces time complexities from exponential to polynomial.


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

## 1.3 Two Pointers 

**11. Container With Most Water - Two Pointer Explanation:**

In the "Container With Most Water" problem, we are asked to find two lines (which we can interpret as array indices), along with the space between them, that can contain the most water. This problem can be efficiently solved by using the two-pointer technique.

The intuition behind the two-pointer approach is the following:

- We start with the maximum width container and every time move the shorter line towards the other end by one step. 

- At the beginning, we can have two pointers, left (starting from the beginning of the array) and right (starting from the end of the array). The area is calculated as the length of the shorter line times the distance between the two pointers.

- On each step, we move the pointer of the shorter line towards the other pointer. The reason behind this is that if we move the pointer of the taller line, we are not sure whether we will get a greater area or not as the width is decreasing, but we know that the height will definitely not increase. On the other hand, by moving the shorter line's pointer, we are hoping that we will find a greater height with less width, leading to a larger area. (虽然不知道面积随宽度的变化，但是在每一个宽度减小时，优先检查最高的高度，也就是当前宽度最大面积。)

This way, the two-pointer technique guarantees to find the optimal solution as it exhaustively checks all possible heights.

**How to Identify Two-Pointer Problems:**

You can generally recognize two-pointer problems by these characteristics:

1. **Sequential Data Structure:** Problems that involve sequential data structures (like arrays or linked lists), where the order of elements is important, often lend themselves well to the two-pointer technique.

2. **Looking for Pairs or Triples:** Many problems that ask you to identify pairs or triples of elements that satisfy certain conditions (like their sum or product equals a certain target value) can often be solved with two pointers.

3. **Searching for a particular sequence or pattern:** If the problem involves searching for a subarray, a subsequence, or any specific pattern in a sequential data structure, two-pointer technique can be useful.

4. **Requires Optimization:** If the problem asks for optimization like finding a maximum or minimum, it's a good hint for possible two-pointer usage.

5. **Involves Forward and/or Backward Traversal:** If the problem involves forward and backward traversal simultaneously in a sequence, it's likely a candidate for two pointers.

**Examples:**

1. **Pair with Given Sum:** Given a sorted array and a target sum, find if there is a pair with the given sum. Here, one pointer starts from the beginning and the other from the end. The pointers move towards each other until they either meet or find the pair with the given sum.

2. **Subarray with Given Sum (Non-negative numbers):** Given an array of non-negative numbers and a target sum, find a subarray with the given sum. Here, two pointers both start from the beginning. The right pointer keeps moving until the sum of elements between two pointers is less than or equal to the target. If the sum is more than the target, the left pointer starts moving until the sum becomes less than or equal to the target.

3. **Remove Duplicates from Sorted Array:** Here, two pointers are used to keep track of the last non-duplicate element and to traverse the array. The pointer traversing the array only advances when a new non-duplicate number is found.

4. **Linked List Cycle Detection (Floyd’s Cycle-Finding Algorithm):** Two pointers move at different speeds to detect a cycle in a linked list. If there is a cycle, the two pointers will eventually meet; otherwise, the faster pointer will reach the end of the list.

5. **Palindrome Linked List:** One pointer moves twice as fast as the other, allowing it to reach the middle of the list while the other is only halfway there. The half-speed pointer also reverses the first half of the list as it goes. When the fast pointer reaches the end, the slow pointer will have reversed half the list and can then compare the reversed first half with the second half for palindrome checking.
   
The two-pointer technique is an essential tool to improve time complexity and often space complexity in many situations, especially when dealing with arrays or linked lists.

## 1.4 Sliding Window

**Sliding Window Technique - Tutorial**

The sliding window technique is commonly used for array and string problems where we need to find a "subarray" or "substring" satisfying certain conditions. The sliding window technique helps in reducing the time complexity.

**1. Understanding Sliding Window:**

Consider an array of integers. A "sliding window" is a subarray that slides from the start of the array to the end. The window size remains constant and is smaller or equal to the size of the array.

**2. How It Works:**

- Identify a suitable window size based on the problem. The window size could be a fixed number or variable that changes throughout the problem.

- Start from the 0th index and slide the window from left to right across the array. You stop sliding when the right side of the window reaches the final element of the array.

- While sliding, keep track of the condition specified in the problem (for example, the sum of elements, maximum/minimum element, etc.).

**3. Characteristics of Sliding Window Problems:**

- The problem will involve a data structure such as a linear array or a string.
- You need to find a subrange that satisfies certain constraints.
- Naive or brute-force solutions usually result in O(n^2) time complexity, but with sliding window, you can bring that down to O(n).

**4. Examples and How to Solve Them:**

*Example 1: Maximum/Minimum Sum Subarray of Size K (fixed window size)*

Given an array of integers and a number K, you have to find the maximum/minimum sum of 'K' consecutive elements in the array.

- The window size here is 'K'.
- Start from the 0th index, calculate the sum of the first 'K' elements.
- Then, slide the window by one position to the right. For the new window, calculate the sum by just adding the new element included in the window and subtracting the first element of the previous window.
- Continue this process until the end of the array and keep track of the maximum/minimum sum.

*Example 2: Longest Substring with K Distinct Characters (variable window size)*

Given a string, find the length of the longest substring with 'K' distinct characters.

- The window size here is variable.
- Start from the 0th index, keep adding characters to the window until it contains 'K' distinct characters.
- If adding a new character makes 'K+1' distinct characters in the window, start from the left to remove characters until you're back to 'K' distinct characters.
- Keep track of the longest window that satisfied this condition.

**5. Benefits of Sliding Window:**

- The sliding window technique reduces time complexity from a brute force O(n^2) to a more efficient O(n).
- It provides a systematic way of navigating through the array or string while keeping track of the problem's requirements.

Sliding window is an elegant technique that can greatly optimize your code, and is very useful for interview problems, competitive programming, and real-world applications.

## 1.5 DFS and BFS

Sure! Let's dive into BFS (Breadth-First Search) and DFS (Depth-First Search) algorithms, which are both fundamental graph traversal techniques used in Computer Science.

**Breadth-First Search (BFS)**

BFS is a graph traversal algorithm that explores all the vertices of a graph in breadth-first order, i.e., it visits vertices that are at the same level before going to the next level. 

**Steps:**

1. Start by visiting a selected node of a graph, marking it as visited or "discovered" and enqueue it into a queue data structure.
   
2. While the queue is not empty:

   - Dequeue a node from the queue and examine it. If the element sought is found in this node, quit the search and return a result. 

   - Otherwise enqueue any successors (the direct child nodes) that have not yet been discovered.
  
3. If the queue is emptied without finding the target, then the target is not present in the graph.

BFS is particularly useful for finding the shortest path in unweighted graphs or solving problems like finding all nodes within one connected component.

**Depth-First Search (DFS)**

DFS is another graph traversal algorithm that explores as far as possible along each branch before retracing steps. It uses a stack (implicitly via recursion, or explicitly) to remember to get back to the nodes.

**Steps:**

1. Start by visiting an arbitrary node (starting node) of a graph, marking it as discovered or "visited" and push it into a stack data structure.
   
2. While the stack is not empty:

   - Pop a node from the stack to select the next node to visit and push all its adjacent nodes into the stack.

   - Continue this process until the stack is empty.
  
DFS is often used for tasks such as connected component discovery, topological sorting, and detecting cycles.

**Identifying When to Use BFS or DFS**

1. **Shortest Path and Simpler Routing**: BFS is better when finding the shortest path in unweighted graphs. BFS will reach the destination vertex in the shortest path in an unweighted graph due to its queue structure. It exhausts all possibilities 'horizontally' before moving on.

2. **Exploring All Options**: DFS, by using a stack structure, will plunge depth-wards into a graph, which can be useful if you want to exhaust all possibilities, or 'drill down' to certain parts of a graph.

3. **Topological Sorting**: DFS can be easily modified to find a topological order of the nodes in a graph. In DFS, we start from a vertex, we first print it and then recursively call DFS for its adjacent vertices. In topological sorting, we use a temporary stack. We don't print the vertex immediately, we first recursively call topological sorting for all its adjacent vertices, then push it to a stack. Finally, print the contents of the stack.

4. **Connected Components**: Both BFS and DFS can be used to find a connected component in a graph. BFS can find the connected component of any given node in O(V+E) time where V and E are vertices and edges respectively.

Sure, here are some examples of LeetCode problems that can be solved using BFS and DFS:

**BFS (Breadth-First Search) Problems:**

1. **Binary Tree Level Order Traversal (LeetCode #102):** This problem asks for the level order traversal of a binary tree, where the nodes at each level are returned as separate lists. This is a perfect use case for BFS since BFS naturally processes nodes level by level.

2. **Word Ladder (LeetCode #127):** In this problem, you have to transform one word into another by changing only one letter at a time, where each intermediate word must exist in a provided dictionary. The problem asks for the minimum length of the transformation sequence, which corresponds to the shortest path in the graph where vertices are words and edges connect words that differ by one letter. This shortest path problem in an unweighted graph is a classic BFS problem.

3. **Open the Lock (LeetCode #752):** In this problem, each state can be represented as a node and transitions between states are edges. We can apply BFS to find the minimum number of moves.

**DFS (Depth-First Search) Problems:**

1. **Number of Islands (LeetCode #200):** This problem asks for the number of islands in a grid where land is represented as 1's and water is represented as 0's. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. We can use DFS to mark each connected component (island) and count the number of such components.

2. **Max Area of Island (LeetCode #695):** Similar to the above problem, but now we need to find the maximum area of an island in the given 2D array. Here, DFS can be used to explore each island and calculate its area.

3. **Generate Parentheses (LeetCode #22):** This problem asks to generate all combinations of well-formed parentheses. DFS (or backtracking, which is essentially DFS) can be used here to explore all possible placements of parentheses.





