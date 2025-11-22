# Introduction to Algorithms 4th Edition

# website: https://learnxinyminutes.com/python/

# 🧠 LeetCode Pattern Recognition + Algorithm & Data Structure Table

A master classification table to help instantly decide **which algorithm or DS to use** for any LeetCode problem.

---

## 🟦 Dynamic Programming (DP)

| Pattern | When to Use | Algorithm | Example Keywords |
|--------|-------------|-----------|------------------|
| 1D DP (Fibonacci style) | Small state, build from previous | dp[i] | "ways", "steps", "min cost", "climb", "decode" |
| 2D DP | Compare two sequences | dp[i][j] | "subsequence", "edit distance", "LCS", "palindrome" |
| Knapsack DP | Choose items with limit | 1D/2D knapsack | "capacity", "max value", "subset sum" |
| DP on Intervals | Solve between i–j | dp[i][j] = best split | "burst balloons", "merge stones" |
| Bitmask DP | DP over subsets | dp[mask][i] | "TSP", "assign workers", "minimum time" |
| Tree DP | DP on children | postorder dp | "tree", "rob", "subtree best" |

---

## 🟧 Greedy Algorithms

| Pattern | When Better Than DP | Greedy Strategy | Keywords |
|--------|----------------------|------------------|----------|
| Interval Scheduling | Choose earliest finishing | Sort by end | "intervals", "merge", "meetings" |
| Activity / Job Scheduling | Max jobs under constraints | Sort by start or profit | "schedule", "jobs", "minimum rooms" |
| Greedy + Heap | Always pick best element | heapq | "k smallest", "k largest", "merge" |
| Greedy for Strings | Lexicographically smallest | stack removal | "remove k digits", "smallest" |

---

## 🟩 Sliding Window + Two Pointers

| Pattern | When to Use | DS | Keywords |
|--------|-------------|----|----------|
| Fixed Window | Constant window size | None | "subarray of size k" |
| Variable Window | Expand/shrink | HashMap | "longest substring", "at most k" |
| Two Pointers | Sorted or monotonic | pointers i,j | "sorted", "pair sum", "remove duplicates" |
| Fast–Slow Pointers | Linked lists | Floyd cycle | "cycle", "middle", "happy number" |

---

## 🟫 Graph Algorithms

| Problem Type | Best Algorithm | When | Keywords |
|--------------|----------------|------|----------|
| Unweighted shortest path | BFS | all edges = 1 | "minimum moves", "steps", "levels" |
| Weighted shortest path | Dijkstra | positive weights | "cheapest", "weighted path" |
| Negative weights | Bellman-Ford | negative edges | "negative", "detect cycle" |
| Cycles in graph | DFS or Union-Find | directed/undirected | "cycle", "dependencies" |
| Topological ordering | Kahn / DFS topo | DAG | "order", "prerequisite" |

---

## 🟪 Trees

| Pattern | Technique | Keywords |
|--------|-----------|----------|
| DFS Tree | in/pre/post-order | "root", "sum", "balanced" |
| Tree Diameter | DFS twice | "diameter", "longest path" |
| LCA | Binary lifting / parent pointers | "lowest common ancestor" |
| BFS Tree | level order | "distance", "depth" |
| Tree DP | combine children results | "rob", "camera", "monitor" |

---

## 🟨 Trie / Prefix Tree

| Pattern | When to Use | Keywords |
|--------|-------------|----------|
| Word dictionary | fast prefix lookup | "startsWith", "search" |
| Multi-word storage | autocomplete | "prefix", "word list" |
| Replace words | replace root forms | "dictionary", "prefix match" |

---

## 🟦 Hashing

| Pattern | Usage | Keywords |
|--------|--------|----------|
| Frequency counting | count items | "anagram", "k frequent" |
| HashSet lookup | O(1) existence | "contains", "duplicate", "membership" |
| Prefix-sum hash | fast sum ranges | "subarray sum", "target sum" |

---

## 🟥 Stack-Based Patterns

| Problem | DS Used | Keywords |
|---------|---------|----------|
| Valid parentheses | stack chars | "valid", "balanced" |
| Monotonic stack | increasing/decreasing | "next greater element", "daily temps" |
| Evaluate expressions | operator stack | "calculate", "expression", "RPN" |

---

## 🟫 Heap / Priority Queue

| Pattern | When | Keywords |
|--------|-------|----------|
| k largest/smallest | maintain top k | "kth", "top k" |
| Scheduling | pick earliest/shortest | "meeting rooms", "tasks" |
| Merge lists | repeatedly take smallest | "merge k lists" |

---

## 🟧 Binary Search

| Type | When to Use | Keywords |
|------|--------------|----------|
| Binary search on array | sorted array | "search", "insert position" |
| Binary search on answer | monotonic condition | "minimum capacity", "smallest days", "feasible" |

---

## 🟦 Math / Bit Manipulation

| Pattern | Use Case | Keywords |
|--------|-----------|----------|
| Bitwise tricks | XOR uniqueness | "single number", "toggle" |
| Counting bits | dp or bit shifts | "count bits" |
| Modular arithmetic | large numbers | "mod", "ways" |
| Number theory | GCD, LCM | "fraction", "simplify" |

---

## 🟩 Range Query Structures

| Problem Type | DS | Keywords |
|--------------|----|----------|
| Range sum update | Fenwick tree | "update + query" |
| Range min/max | Segment tree | "range minimum", "interval" |
| Interval addition | difference array | "range add", "multiple updates" |

---

## 🟥 Backtracking

| Pattern | Use Case | Keywords |
|--------|-----------|----------|
| Subsets | choose/not choose | "subset", "power set" |
| Permutations | reorder all | "permutation", "all arrangements" |
| Combinations | choose k | "combination", "sum to target" |
| N Queens / Sudoku | constraints | "valid placements", "fill board" |

---

## 🟧 Strings

| Pattern | Technique | Keywords |
|--------|------------|----------|
| Expand around center | palindromes | "longest palindrome" |
| Two pointers | reverse, trim | "reverse words", "remove chars" |
| Hashing | rolling hash | "substring equal", "find pattern" |
| DP | edit distance | "distance", "alignment" |

---

## 🟦 Intervals

| Pattern | Technique | Keywords |
|--------|-----------|----------|
| Merge intervals | sort + merge | "intervals", "overlap" |
| Insert interval | binary search + merge | "insert", "overlapping" |
| Meeting rooms | min heap | "rooms", "availability" |

---

## 🟫 Arrays & Prefix-Sum

| Pattern | Technique | Keywords |
|--------|-----------|----------|
| Prefix sum | cumulative | "range sum", "subarray = target" |
| Difference array | fast update | "multiple updates", "range add" |
| Kadane's algorithm | max subarray | "maximum subarray", "largest sum" |

---

## 🟨 Matrix Problems

| Pattern | Technique | Keywords |
|--------|-----------|----------|
| Flood fill | DFS/BFS | "islands", "regions" |
| DP on matrix | bottom-up | "min path sum", "unique paths" |
| BFS shortest path | 0-1 BFS for weights | "shortest path", "grid" |

---

## 🟪 Linked Lists

| Pattern | Technique | Keywords |
|--------|------------|----------|
| Reverse list | pointers | "reverse", "swap nodes" |
| Cycle detection | Floyd's tortoise-hare | "cycle", "loop", "detect" |
| Merge lists | dummy head + pointers | "merge", "sort list" |

---

## 🟧 Union-Find (Disjoint Set)

| Pattern | Use Case | Keywords |
|--------|-----------|----------|
| Connected components | find/union | "connected", "groups", "components" |
| Cycle detection | union | "cycle", "redundant edge" |
| Kruskal MST | sort edges + DSU | "minimum spanning tree" |

---

## 🟥 Sorting Patterns

| Pattern | Use Case | Keywords |
|--------|-----------|----------|
| Sort by custom key | define lambda | "sort by frequency", "sort by value" |
| Bucket sort | limited range | "top frequencies", "group" |
| Counting sort | small ranges | "age sort", "score sort" |

---

# ✔️ Quick Problem → Algorithm Lookup

| If it asks for… | Use |
|------------------|-----|
| Longest substring | sliding window |
| Minimum cost path in matrix | DP or Dijkstra |
| All permutations | Backtracking |
| K largest elements | Heap |
| Next greater element | Monotonic stack |
| Number of islands | BFS/DFS |
| Cheapest path | Dijkstra |
| Precedence / order | Topological sort |
| Detect cycle | DFS or Union-Find |
| Range updates | Fenwick / Segment Tree |
| Palindromic substring | expand around center |

---

# END OF MARKDOWN
