
# BFS and DFS
## BFS
### BFS in tree
- Binary Tree Level Order Traversal (102)
- Binary Tree Right Side View (199)
- Average of Levels in Binary Tree (637)
- Binary Tree Zigzag Level Order Traversal (103)
- Recover BST () inorder traverse should be ascending, find the 2 that are not and switch

### BFS in graph
- 01 Matrix (542)
- snakes and ladders (Non-linear not solvable by DP)

## DFS
### DFS in graph
- Number of Islands (200)
- Surrounded Regions (130)
- * Word Search (Typical DFS) *
- course schedule I (207)

*  To do *
- course schedule II (207 210)
-  Max Area of Island (695)
-  Generate Parentheses (22)
-  Clone Graph  (133)

# Dynamic Programming
- Fibonacci numbers (509) dp better than recursion. recursion abundant computing O(2^n)
- Packaging problems (1049)
- Climbing stairs (70,746)
- Jump Games (45, 55)
- Frog Jump (403)
- Edit Distance ()
- Interleaving String () *review*
- longest increasing subsequence *review*
  
## graph

- Unique Paths I and II (62,63)
- Maximal Square (221)

## max diffenrence
- best time to sell stocks I to V (121,122,123,188,309,714)
- largest temperature difference.

## subset
- Palindromic Substrings(647)
* - Longest Palindromic Subsequence (516) *

# Backtrack !! review more

- Combinations (77)
- Permutations (46)
- Combination sum (39)
- Letter Combinations of a Phone Number (17)
- Partitioning Panlindrome (131)
- Leetcode 332 (Reconstruct Itinerary)
- Generate Parentheses (22)
  

# Two Pointers
- two sum , three sum (remember to avoid duplicated cases) (review to see how to use set properly)
- contain most water
- trapping rain water (can also use dp)
- Remove Nth Node From End of List 19 (fast slow pointer. let fast move n step first. similar to use 15min and 1 hour candle to get 45min. use dummy head to avoid specially dealing with deleting first node)

# Sliding Window
- gas station
- merge intervals (compare the start and end, add one by one to res [])
- longest substring without repeating characters. (use character set to record unique characters in current window.) set is effcient.

# linked list
- reverse linked list
- Reverse Linked List II (92)
- reverse group list
- Add Two Numbers (2) (remember how to value list node ans.next = ListNode(sum_digit))

# Priority Queues (Heap)
- Divide Intervals Into Minimum Number of Groups (2406)
- The Number of the Smallest Unoccupied Chair(1942)
- Meeting Rooms II (253)
- Longest happy numbers 1405 (use heap to get max_count of abc, priority queue to use max_count character)

# Hash tables
Group Anagrams (49) (Use defaultdict(list) to avoid key error / use sorted(s) as shared key)

# String
- Reverse Integer (string operation, use[::-1] for reverse. compare every digit with the 32digit max)
- compare version number
- decode string         # use stack to get everything in [] # plain case : no nested, everything inside * num # nest case: recursively call this funciton

# prefix
Maximum Subarray (53)

# Interval
Merge Interval
Insert Interval


# Tiktok
## unsolved 
### from zhihu https://zhuanlan.zhihu.com/p/392284340


-  Search in Rotated Sorted Array (33)


-  Permutation Sequence (60)
- 判断是否存在个数超过数组长度一半的数

- 翻转带环的链表



### from interview experiences(Niu ke) AI-lab
-  Wildcard Matching (44)
-  Lowest Common Ancestor of a Binary Tree

- intersection-of-two-linked-lists (160) hashmap

### from leetcode Tiktok collections
- Longest Substring with At Least K Repeating Characters
- Merge Intervals
- Two City Scheduling




## NOT fimiliar
- subarray sum equals to k (560) store every sum in hashmap, if sum - k seen , from there to here is a solution.  
- Nth digit (400) 1. find digit_len, 2. find num by (//) 3. find digit by (%).
- K-th Smallest in Lexicographical Order () 10-ary tree
- Kth Largest Element in an Array (215) quick select.

- longest-increasing-path-in-a-matrix (329) dfs + dp save the visited cell
- linked list cycle I and II (141 and 142) slow fast pointer
- Median of Two Sorted Arrays (4)
  


- Course Schedule () dfs detect cycle topological sorting
- Edit Distance dp transition from (delete, insert and replace)



## Solved 
## graph traverse
- Spiral Matrix () traverse ,if outside of visited , undo and change direction.

### binary tree 
- Binary Tree Maximum Path Sum (124) recursion
- Path Sum I and II recursion/dfs

### dfs
- Word Search () dfs
- 01 Matrix () dfs
- Number of Island () dfs 
- Max area of island () dfs

- Search a 2D Matrix I -- flatten as 1D.
- Search a 2D Matrix II  -- start from top-right corner, < move down, > move left.


- gas station () sliding window

### dp
- Unique Path I and II () dp
- Minimum Path Sum (64) Simple dp , use grid iteself as dp.
 

## tricky (math)
-  动物园有猴山，每天需要给猴子们发香蕉,猴子会排队依次取食。 猴子们铺张浪费,会多拿食物,但最多不会拿超过自身食量的二倍且不会超过当前还存在的香蕉的一半,最后—个猴子除外(即最后—个猴子可以拿完剩余的所有香蕉)。 最少需要准备多少香蕉,能保证所有猴子都能吃饱? 输入每个猴子的食量，输出最少的香蕉个数 —————— 逆向思维，从最后一个猴子开始，排序猴子的食量从大到小，贪心算法推到最第一个。

```
  def min_bananas(monkeys):
      monkeys.sort(reverse=True)
      bananas = monkeys[0]
      for i in range(1, len(monkeys)):
          bananas = max(2 * monkeys[i], bananas + bananas // 2)
      return bananas
```


- Koko Eating Bananas (875) binary search , ceil()

- 现在有一堆点，求一个点到每个点的距离之和最小，证明这个点是质心。
- 甲扔n次骰子，取其中最大的点数作为它的最终点数，乙扔一次骰子得到点数，求乙的点数大于甲的概率。
- 某种病的发病率为1/100，某种检测该病的技术检测正确率为99/100，现有一人被检测到生病的概率为p，求他真实生病的概率是多少？
- 在上一问的基础上，现在连续两次检测为有病才会停止检测，求检测次数的期望值。
- 概率题， 10个人里每个人在10分钟内的任何一个分钟到达的概率是均匀分布的，问所有人都到达的时刻在几分钟时概率最大。

### deep learning questions
- 怎么解决梯度消失问题？
- 批量归一化的思想，还了解其他归一化吗？
- 说下平时用到的深度学习的trick
- 说下adam的思想

## First Round
- course schedule
- rotate image
## second round 
- vertical of binary tree
## third round
- longest subsequence of consecutive natural number

## First round (backend)
- number of ways to decode (91)
- trim binary search tree (669)

## Second round



Slow fast pointer theory 

Definitions:

Let's assume the distance from the head of the linked list to the start of the cycle is a.
The distance from the start of the cycle to the point where the slow and fast pointers first meet is b.
The remaining distance of the cycle (from the meeting point back to the start of the cycle) is c.
Distances Covered:

When the slow pointer and fast pointer meet, the slow pointer has traveled a distance of a + b.
The fast pointer has traveled a distance of a + b + c + b (it has covered the cycle more than once).
Since the fast pointer travels at twice the speed of the slow pointer, its distance is also twice that of the slow pointer: 2(a + b).
Equating the Distances:

From the above, we can write the equation: 2(a + b) = a + b + c + b.
Simplifying, we get: a = c.
Interpretation:

The distance a (from the head to the start of the cycle) is the same as the distance c (from the meeting point of the pointers back to the start of the cycle).
This means that if we reset the slow pointer to the head and move both pointers one step at a time, they will meet at the start of the cycle. This is because the slow pointer will cover a distance of a to reach the start of the cycle, and the fast pointer (starting from the meeting point) will also cover a distance of a (which is equivalent to c) to reach the start of the cycle.
This mathematical insight is the reason why resetting the slow pointer to the head and then moving both pointers one step at a time until they meet again gives the start of the cycle.








1049. Last Stone Weight II:

Problem: You are given an array of integers stones where stones[i] is the weight of the i-th stone. We are to choose the largest two stones and smash them together. If the stones have the same weight, both stones are destroyed; if not, the stone with the smaller weight is destroyed, and the other stone's weight is reduced by the smaller stone's weight. The problem is to find the smallest possible weight of the last stone remaining.
Approach: This problem can be reduced to a subset sum problem and can be solved using dynamic programming.
743. Network Delay Time:

Problem: There are N network nodes, labeled from 1 to N. Given times, a list of travel times as directed edges times[i] = (u, v, w), where u is the source node, v is the target node, and w is the time it takes for a signal to travel from source to target. We send a signal from a given node K. Return the time it takes for all nodes to receive the signal. If not all nodes receive the signal, return -1.
Approach: This is a shortest path problem that can be solved using Dijkstra's algorithm or BFS.
787. Cheapest Flights Within K Stops:

Problem: Given the flight itinerary consisting of starting city A, destination city B, and the cost of the ticket, along with a maximum number of stops K, find the cheapest price from A to B with at most K stops. If there is no such route, output -1.
Approach: This problem can be approached using BFS or dynamic programming.
332. Reconstruct Itinerary:

Problem: Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.
Approach: This problem can be solved using DFS.
1202. Smallest String With Swaps:

Problem: You are given a string s, and an array of pairs of indices in the string pairs where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string. You can swap the characters at any pair of indices in the given pairs any number of times. Return the lexicographically smallest string that s can be changed to after using the swaps.
Approach: This problem can be approached using union-find or DFS.
These problems involve undirected or directed graphs with some cost or weight and can be tackled using graph traversal techniques like DFS, BFS, or optimization techniques like dynamic programming.






