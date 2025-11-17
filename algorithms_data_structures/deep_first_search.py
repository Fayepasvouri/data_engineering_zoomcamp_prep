# deep first search is a graph from smallest left to biggest right and HEAD vertex. it has vertices and edges and its Big O complexity is O(V + E)

# Simple implementation

def deep_first_search(graph, start, visited=None):
    
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    for next_val in graph[start] - visited:
        deep_first_search(graph, next_val, visited) # dfs uses recursion
    print(visited)
    return visited

graph = {
    1: set([2, 3]),
    2: set([4, 5]),
    3: set([6, 7]),
    4: set([]),
    5: set([]),
    6: set([]),
    7: set([])
}

deep_first_search(graph, 1)

# 104 leetcode easy find depth of a tree

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        left = maxDepth(root.left) # index size 0, 1 etc .. recursion is required in DFS
        right = maxDepth(root.right)
        return 1 + max(left, right) # index size + 1 to find depth
# 111 leetcode easy 

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0 
        
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if left == 0:
            return 1 + right
        elif right == 0:
            return 1 + left
        return 1 + min(left, right)
