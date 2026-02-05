
import heapq
from collections import deque
from typing import List, Optional


class Node:
    '''
    Basic Node, stores pointer to val and next 
    Also contains pointer for left/right if needed
    '''    
    def __init__(self, val=None, prev=None, next = None, left=None, right=None, parent = None):
        self.value = val
        self.next = next
        self.prev = prev
        self.left = left 
        self.right = right 
        self.parent = parent

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def python_dictionary():
    '''
    Dictionary/Hashmap
    '''
    print("\n====== Dictionary ======")

    # Decleration
    p_dict: dict = {}
    p_dict = {'foo': 4098, 'bar': 4139, 'toremove': 0}

    # Add/Update for a given key
    p_dict["key"] = "value"

    # Accessing
    item = p_dict["key"]
    print(f"Access valid item: {item}")
    non_exist = p_dict.get("unset")
    print(f"Access valid item: {non_exist}")

    # Removing Key if exists
    del p_dict['toremove']

    # Iterate Keys
    print(f"Printing Keys") 
    for key in p_dict.keys():
        print(key)

    # Iterate Values
    print(f"Printing Values") 
    for val in p_dict.values():
        print(val)

    # Iterate KVPs at once 
    print(f"Printing KVPs") 
    for k,v in p_dict.items():
        print(f"key: {k}, val: {v}")
    
    print("====== Dictionary ======")
    return

def python_hashset():
    '''
    HashSet
    '''
    print("\n====== Set ======")

    # Decleration
    p_set: set = set()
    p_set = {"foo", "bar"}

    # Add
    p_set.add("key")

    # Remove
    p_set.remove("key")
    # Remove even if not in set
    p_set.discard("key")

    # Access (In set)
    print(f"Is 'foo' in set: {'foo' in p_set}")
    print(f"Is 'baz' in set: {'baz' in p_set}")

    print("Printing Set Items:")
    for item in p_set:
        print(item)

    print("====== Set ======")
    return

def python_stack():
    '''
    Stack
    Uses List declaration
    '''
    print("\n====== Stack ======")

    # Decleration
    p_stack = []
    p_stack = ["foo", "bar"]

    # Push Element
    p_stack.append("key")

    # Peek Element 
    peek = p_stack[-1]
    print(f"Top of stack peeked: {peek}")

    # Pop
    item = p_stack.pop()
    print(f"Popped stack item: {item}")

    # Is empty 
    is_empty = not bool(p_stack)
    print(f"Is Stack empty: {is_empty}")

    # Size
    print(f"Stack Size: {len(p_stack)}")
    print("====== Stack ======")
    return

class MyStack():
    '''
    A Linked List implementation of a Stack
    '''
    def __init__(self):
        self.head: Node = None 
        self.size = 0

    def push(self, value):
        node = Node(val=value)
        if self.head:
            node.next = self.head
        self.head = node
        self.size += 1

    def peek(self):
        if self.is_empty():
            return None
        return self.head.value

    def pop(self):
        if self.is_empty():
            return None
        val = self.head.value
        self.head = self.head.next
        self.size -= 1
        return val

    def is_empty(self):
        return self.size == 0

    def stack_size(self):
        return self.size

    def traverse_print(self):
        curr = self.head
        while curr:
            print(curr.value, end="->")
            curr = curr.next 
        print("\n")


def python_queue():
    '''
    Queue, Python internals is a dequeue
    '''
    print("\n====== Queue ======")

    # Decleration
    p_q: deque = deque()
    
    # Enqueue 
    p_q.append('foo')
    for n in range(3):
        p_q.append(f"s{n}")
    print("printing items in queue")
    for item in p_q:
        print(item)

    # Dequeue AKA remove from front 
    item = p_q.popleft()
    print(f"Dequeued item: {item}")

    # Peek (Q is array based and index 0 is "front")
    front = p_q[0]
    print(f"Q start/peek item: {front}")

    # Check End 
    end = p_q[len(p_q)-1]
    print(f"Q end item: {end}")

    is_empty = len(p_q) == 0
    print(f"Is q empty: {is_empty}")

    print("====== Queue ======")
    return

def python_heapq():
    '''
    Heap Queue or Priority Queue
    '''
    print("\n====== Priority Queue ======")

    # Decleration, min heap by default
    p_pq: heapq = []
    
    pq_items = {
        (1, "foo"), # highest 
        (5, "bar"), # low 
        (3, "baz")  # med
    }
    # Enqueue with priority (Priority, Value)
    heapq.heappush(p_pq, (9, 20)) # lowest prio
    for item in pq_items:
        heapq.heappush(p_pq, item)
    print("PQ contents:", p_pq)

    # Dequeue (remove highest priority = lowest number)
    priority, item = heapq.heappop(p_pq)
    print(f"Priority Queue Pop: prio: {priority}, item: {item}")

    # Peek (look at highest priority without removing)
    priority, item = p_pq[0]  # Doesn't remove
    print(f"Priority Queue Peek: prio: {priority}, item: {item}")

    # Check if empty
    is_empty = len(p_pq) == 0
    print(f"Is PQ empty: {is_empty}")

    print("====== Priority Queue ======")
    return

def python_string():
    print("\n====== Strings ======")

    p_str: str = "strings"

    for c in p_str:
        print(c)

    print("====== Strings ======")
    return


def python_sorting():
    '''
    Builtins Sorting 
    '''
    # NOTE sets cannot be sorted. always use a list
    unsorted_list = [5, 2, 8, 3, 9]
    print(f"Unsorted: {unsorted_list}")
    
    # copy sorted
    sorted_copy = sorted(unsorted_list)
    print(f"Copy Sorted: {sorted_copy}")

    # In place sort
    unsorted_list.sort()
    print(f"Sorted Initial: {unsorted_list}")

def python_loops():
    '''
    looping strats
    '''
    n = 4
    # for (int i = 0; i < n; i+=1)
    for k in range(0, n, 1): # arg0 is start inclusive, arg1 is end exclusive, arg3 is iterate size
        print(k)
    # for (int i = n; i > 0; i-=2)
    for i in range(n, 0, -2):
        print(i)

# ========= Array Manipulation =========
# Prefix Suffix Sums
def productExceptSelf(self, nums: List[int]) -> List[int]:
    # https://leetcode.com/problems/product-of-array-except-self/
    '''
        since no division and O(n) limitation, we know that when scanning the array
        we can get the value of the current index since by the left and right intervals
        i = [left] * [right]
        approach: build the prefix and suffix arrays, then mult pref and suff to get correct i
    '''
    n = len(nums)

    prefix = [0] * n # prefix[i] will contain all the current values to the left
    suffix = [0] * n # suffix[i] contains all prods to the right
    result = [0] * n 
    
    prefix[0] = 1
    for i in range(1, n):
        prefix[i] = nums[i-1] * prefix[i-1]
    suffix[n-1] = 1
    for i in range(n-2, -1, -1):
        suffix[i] = nums[i+1] * suffix[i+1]

    for i in range(n):
        result[i] = prefix[i] * suffix[i]

    return result

def isValidSudoku(self, board: List[List[str]]) -> bool:
    # https://leetcode.com/problems/valid-sudoku/
    for i in range(9):
        rset = set()
        cset = set()
        gset = set()
        for j in range(9):
            row = board[i][j]
            col = board[j][i]

            if row != ".":
                if row in rset:
                    return False
                rset.add(row)

            if col != ".":
                if col in cset:
                    return False
                cset.add(col)

            # also check the 3x3 grids to make sure they hold up the rule 
            gr = 3 * (i//3) + (j//3)
            gc = 3 * (i%3) + (j%3)
            grid = board[gr][gc]
            if grid != ".":
                if grid in gset:
                    return False
                gset.add(grid)
    return True


# ========= Array Manipulation =========


# ========= Frequency Map =========
def frequency_map(values:st[any]) -> dict:
    # Use Dict where value of keys is its occurance 
    freq:dict = {} 
    for item in values:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def topKFrequent(nums: List[int], k: int):
    # https://leetcode.com/problems/top-k-frequent-elements/
    # Generate freq map 
    fmap = frequency_map(nums)
    # Generate Buckets for frequencies, remapping this bucket where 
    # the index represets numbers of that occurance (i=1, 1 occurance)
    b_size = len(nums) + 1 
    buckets = [[] for _ in range (b_size)] # init an array of empty arrays
    for key, _ in fmap.items():
        occ = fmap[key]
        buckets[occ].append(key)
    # build the resulting array of k length
    result = [0] * k
    index = 0
    # iterate backwards of the bucket to retrieve top k
    for bucket in reversed(buckets):
        if len(bucket) > 0:
            for item in bucket:
                # NOTE it retrieves the order in which the buckets were generated, in ties the first item found is whats added
                result[index] = item # add all the values to final result
                index += 1
                if index == k:
                    return result
    return result

def reorganizeString(s: str)-> str:
    # https://leetcode.com/problems/reorganize-string/
    # Create a frequency map then iterate through the freqmap and reconstruct
    # this is optimal up to O(n) if the alphabet of the input string is bound (EG 26 a-z) 
    # Otherwise sorting the frequency map will costs O(klogk) k = size of alphabet 
    fmap = frequency_map(s)
    # check if most freq is within at least half of the array, if greater not possible
    if max(fmap.values()) > (len(s) + 1)//2:
        return "" 

    # sort the kvp dict by its value in descending order. This gives us the alphabet in order to rebuild from 
    sorted_chars = sorted(fmap.items(), key=lambda x: -x[1])

    result = [''] * len(s) 
    index = 0 
    # Insert highest frequency on even position first, then swap to odd 
    for char, count in sorted_chars:
        for _ in range(count):
            if index >= len(s):
                index = 1 # switch to odd 
            result[index] = char
            # itaerate on even positions
            index += 2
    
    return ''.join(result)
# ========= Frequency Map =========

# ========= Two Pointers =========
def isPalindrome(s: str) -> bool:
    # https://leetcode.com/problems/valid-palindrome/
    left = 0 
    right = len(s) - 1
    while left < right:
        # check for non alphanumerics
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        # move the pointers naturally
        left += 1
        right -= 1
    return True # reached mid return true

def twoSum_two(numbers: List[int], target: int) -> List[int]:
    # https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
    # Condition array is sorted, finding numbers that add up to target. T = L + R 
    left = 0 
    right = len(numbers) - 1
    while left < right:
        current = numbers[left] + numbers[right] # calculate target and compare
        if current == target:
            return [left + 1, right + 1] # 1 based index from original problem
        # didn't find chose direction to move
        if current > target:
            right -= 1
        else:
            left += 1
    return [-1, -1] # default, didnt find target

def moveZeroes(nums: List[int]) -> None:
    # https://leetcode.com/problems/move-zeroes/description/
    # Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements. INPLACe
    # invert the problem, move NON zeroes to teh front for O(n)
    left = 0 # Used where to insert non-zeroes to
    # right also starts at 0 which is the swapping pointer
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left] = nums[right]
            left += 1 
    # Backfill the array with zeroes starting from the current pos of left
    for i in range(left, len(nums)):
        nums[i] = 0 

def minCost(colors: str, needed_time: List[int]) -> int:
    # https://leetcode.com/problems/minimum-time-to-make-rope-colorful/
    # use sliding window to find adjacent to pop (make colorful)  doing pairwise comparison. Can guarantee minimal by chosing the lowest value
    result = 0
    left = 0
    for right in range(1, len(colors)): 
        # found two adjacent colors, need to pop
        if colors[left] == colors[right]:
            # pick which to pop with less cost
            if needed_time[left] < needed_time[right]:
                result += needed_time[left] # left lower cost pick and pop
                left = right # move left to unpopped 
            else:
                result += needed_time[right]
        else:
            # no same color move left to right's position
            left = right 
    return result
# ========= Two Pointers =========

# ========= Stack =========
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    # https://leetcode.com/problems/daily-temperatures/
    # memorize but possibly soloable
    n = len(temperatures)
    result = [0] * n 
    # stack to hold indices of days still looking for wamer days, monotonic decreasing
    # eg: temps[s[0]] > temps[s[1]] > temps[s[2]] > ... older entries warmer 
    stack = [] 
    for i in range(n):
        temp = temperatures[i]
        while stack:
            prev = stack[-1] # peek stack and check last prev day are warmer than current temp
            if temperatures[prev] >= temperatures[i]:
                break # previous days are warmer, break and preserve monotonic
            # today's day is warmer, continue updating and resolve  
            stack.pop()
            result[prev] = i - prev # trick, since we store indices resolving down stack we just subtract to get the correct wait days 
        stack.append(i) # push current day to check future warmere dates
    return result

def evalRPN(self, tokens: List[str]) -> int:
    # https://leetcode.com/problems/evaluate-reverse-polish-notation/
    stack = [] 
    ops = {'+', '-', '*', '/'}
    for token in tokens:
        if token not in ops:
            stack.append(int(token))
        else:
            # grop 2 nums from stack and apply ops
            b = stack.pop() # the front of the stack, rhs most val
            a = stack.pop() # the older value
            if token == '+':
                stack.append(a+b)
            elif token == '-':
                stack.append(a-b)
            elif token == '*':
                stack.append(a*b)
            elif token == '/':
                stack.append(int(a/b))
    return stack.pop() # evad valuei s front of stack

# ========= Stack =========




# ========= Sliding Window =========

# ========= Sliding Window =========


# ========= Binary Search =========
def binary_search(nums: List[int], target: int) -> int:
    # https://leetcode.com/problems/binary-search/
    left = 0 
    right = len(nums)-1
    while left <= right:
        mid = left + ((right-left)//2) # midpoint overflow protection
        if nums[mid] == target:
            return mid
        elif nums[mid] < target: # go right  
            left = mid+1
        else:
            right = mid-1
    return -1 # default 

def search_rotated_array(nums: List[int], target: int) -> int:
    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    left = 0
    right = len(nums)-1
    while left <= right:
        mid = left + ((right-left)//2)
        if nums[mid] == target:
            return mid
        # find the correct search window partitioned from the rotation
        # attempt to search the left half and see if its sorted
        if nums[left] <= nums[mid]:
            # is the rightmost value greater than target -> must exist in the left half - rotation
            # target is less than mid, then still search the left side
            if nums[left] <= target and target < nums[mid]:
                right = mid - 1 # search the left half
            else:
                left = mid + 1 
        else:
            # right half is sorted, find the right cases to search in 
            if nums[mid] < target and target <= nums[right]:
                left = mid + 1 # target in the right half
            else:
                right = mid -1
    return -1

def isPerfectSquare(self, num: int) -> bool:
    # https://leetcode.com/problems/valid-perfect-square/
    # monotonic values of squares binary search
    if num < 2: 
        return True
    left, right = 1, num // 2 # start searching the half of the num as its sq
    while left <= right:
        mid = left + ((right-left) // 2)
        square = mid*mid
        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid -1
    return False

def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    # indexing : matrix[row][col]
    rows, cols = len(matrix), len(matrix[0])
    i_left,i_right = 0, rows-1
    while i_left <= i_right:
        i_mid = i_left + ((i_right-i_left)//2) # get the middle interval 
        # check if target within this interval and bsearch, otherwise update mid
        if matrix[i_mid][0] <= target and matrix[i_mid][cols-1] >= target:
            a_left, a_right = 0, cols-1
            while a_left <= a_right:
                a_mid = a_left + ((a_right-a_left)//2)
                if target == matrix[i_mid][a_mid]:
                    return True
                elif matrix[i_mid][a_mid] < target:
                    a_left = a_mid+1
                else:
                    a_right = a_mid-1
            return False # could not find in this interval
        elif matrix[i_mid][cols-1] < target:
            i_left = i_mid+1
        else:
            i_right = i_mid-1
    return False # unable to find
# ========= Binary Search =========


# ========= BFS =========
# BFS -> Using a q to track visited
def numIslands(grid: List[List[str]]) -> int:
    # https://leetcode.com/problems/number-of-islands/
    # Also same problem https://leetcode.com/problems/max-area-of-island/
    # Assuming we can modify the graph, we can use it to track visited tiles
    islands = 0 
    rows = len(grid)
    cols = len(grid[0])

    def BFS(r:int, c: int):
        pair = (r,c)
        q = deque()
        q.append(pair)
        while len(q) > 0:
            current = q.popleft()
            row = current[0]
            col = current[1]
            directions = [[0,1], [0,-1], [1,0], [-1,0]] # E W N S
            for direction in directions:
                # create directions to visit
                v_row = row+direction[0]
                v_col = col+direction[1]
                
                # safety bounds check 
                if v_row >= 0 and v_row < rows and v_col >= 0 and v_col < cols: 
                    if grid[v_row][v_col] == '1': #unvisited tile
                        q.append((v_row, v_col)) 
                        grid[v_row][v_col] = '3' # mark as visited

    # traverse grid 
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '3': # visited 
                continue
            if grid[r][c] == '1': # new island, begin BFS
                islands += 1
                BFS(r,c)

    return islands

def invertTree(root: Optional[Node]) -> Optional[Node]:
    q = deque()
    if root:
        q.append(root)

    while len(q) > 0:
        n = q.popleft()
        temp = n.left 
        n.left = n.right
        n.right = temp

        if n.left: 
            q.append(n.left)
        if n.right:
            q.append(n.right)

    return root

def maxDepth(self, root: Optional[Node]) -> int:
    # https://leetcode.com/problems/maximum-depth-of-binary-tree/
    # BFS recursive max. Level Order traversal recurse adding the current depth of the tree 
    if not root:
        return 0 # base case
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) 

def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    # https://leetcode.com/problems/binary-tree-level-order-traversal/
    # Level Order: classical top to bottom BFS, this is tricky medium
    if not root:
        return []
    q = deque([root])
    result = []
    while q:
        level = []
        level_size = len(q) # how many nodes at this level
        for _ in range(level_size): # append the list with ALL traversed nodes of this level
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
# ========= BFS =========


# ========= DFS =========
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    # https://leetcode.com/problems/max-area-of-island/
    # DFS as helper to track visited nodes, marking previously visited nodes
    rows = len(grid)
    cols = len(grid[0])

    # DFS is used to calculate the area of the currently visited island
    def DFS(grid, r, c):
        # bounds check
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 0
        if grid[r][c] == 0:
            return 0
        # mark this node as visited, simply turning it 0 to not need to handle previous code
        grid[r][c] = 0
        area = 1
        # DFS each direction and calculate possible max area
        area += DFS(grid, r+1, c)
        area += DFS(grid, r-1, c)
        area += DFS(grid, r, c+1)
        area += DFS(grid, r, c-1)
        return area

     
    max_area = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(DFS(grid, r, c), max_area)

    return max_area


def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # https://leetcode.com/problems/same-tree/solutions/
    if p is None and q is None: # recursed into empty, still same so its fine
        return True
    # one of the p or q is different (one is None other isnt), must be false
    if p is None or q is None:
        return False
    if p.val == q.val: # compare existing values
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right) 
    return False # previous conditions fail, then there is a difference in the tree

# ========= DFS =========



# ========= Linked List =========
def reverseList(head: Node) -> Optional[Node]:
    # https://leetcode.com/problems/reverse-linked-list/
    prev = None 
    while head:
        next = head.next
        head.next = prev 
        prev = head 
        head = next 
    return prev

def mergeTwoLists(list1: Node, list2: Node) -> Optional[Node]:
    # https://leetcode.com/problems/merge-two-sorted-lists/
    # base cases
    if list1 == None: return list2
    if list2 == None: return list1 

    result = Node(0) # dummy node to being new list
    current = result

    while list1 and list2:
        # pick which sub list to merge its node
        if list1.value <= list2.value:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2 
            list2 = list2.next
        # move current morwards 
        current = current.next
    # insert any remaining nodes as while loop terminates once one is empty 
    if list1:
        current.next = list1
    else:
        current.next = list2
    return result.next # return the merged lists head from the dummy node

# ======== Slow Fast
def hasCycle(self, head: Optional[ListNode]) -> bool:
    # https://leetcode.com/problems/linked-list-cycle/
    # slow fast pointers 
    if not head or not head.next:
        return False
    
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def removeNthFromEnd(self, head: Optional[Node], n: int) -> Optional[Node]:
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    # 1 pass, use slow fast pointers by pushing the fast pointer with a distance k from the slow pointer
    dummy = Node(0)
    dummy.next = head
    slow = dummy 
    fast = dummy 

    # push fast n + 1 gap, so that slow will point to n-1 when fast reaches end
    for _ in range(n + 1):
        fast = fast.next

    # move fast to end with slow moving as well
    while fast:
        fast = fast.next
        slow = slow.next

    # slow now points to n-1, remove it 
    slow.next = slow.next.next
    return dummy.next 

# ======== Slow Fast

#==== helpers
def build_linked_list(values):
    if not values:
        return None

    head = Node(values[0])
    curr = head
    for v in values[1:]:
        curr.next = Node(v)
        curr = curr.next
    return head
def print_linked_list(head):
    vals = []
    while head:
        vals.append(str(head.value))
        head = head.next
    print(" -> ".join(vals))

# ========= Linked List =========

# ========= Memoization =========
def fib(self, n: int) -> int:
    # https://leetcode.com/problems/fibonacci-number/
    # build a cache for prebiously calculated values
    if n <= 1:
        return n
    cache = {} # memoize, store the previously calculated 
    def calc(n):
        if n <= 1:
            return n
        if n in cache:
            return cache[n]
        cache[n] = calc(n-1) + calc(n-2)
        return cache[n]
    return calc(n-1) + calc(n-2)
# ========= Memoization =========


if __name__ == "__main__":

    def ds():
        python_dictionary()
        python_hashset()
        python_stack()

        print("\n====== Linked List Stack ======")
        stack = MyStack()
        for i in range(5):
            stack.push(f"(s:{i})")
        stack.traverse_print()
        print("====== Linked List Stack ======")

        python_queue()
        python_heapq()
        python_string()
        python_sorting()

    # ds()

    # LC 
    k = 3
    d = [5, 8, 3, 9, 9, 9, 9, 9, 0,0,0, 1, 1, 2, 2, 32]
    res = topKFrequent(d, k)
    # print (f"Top K({k}) of {d}:\n{res}")

    # palindrome
    str1 = "racecar"
    str2 = "tomato"
    str3 = "tal, i L At..."

    # print(f"Is Palindrome '{str1}': {isPalindrome(str1)}")
    # print(f"Is Palindrome '{str2}': {isPalindrome(str2)}")
    # print(f"Is Palindrome '{str3}': {isPalindrome(str3)}")

    # 2sum2 
    ts2 = [-9, -2, 2,7,11,15]
    # print(f"Two sum two of: {ts2}, target: {9}\nresult: {twoSum_two(target=9, numbers=ts2)}")
    # print(f"Two sum two of: {ts2}, target: {23}\nresult: {twoSum_two(target=23, numbers=ts2)}")

    # move zeroes 
    mz_nums = [0,1,0,3,12]
    # print(f"Move zeroes of: {mz_nums}")
    moveZeroes(mz_nums) # Modifies in-place
    # print(f"After move: {mz_nums}")

    # min cost
    mc = "abaac" 
    neededTime = [1,2,3,4,5] # answer is 3 
    # print(f"Min cost to make: '{mc}' colorful,  where time: {neededTime}. cost: {minCost(mc, neededTime)}")

    # reorganizeString
    s = "aab"
    print(f"Reorganize String of '{s}':\nresult: {reorganizeString(s)}" )

    # linked list 
    # Build list: 1 -> 2 -> 3 -> 4
    head = build_linked_list([1, 2, 3, 4])
    print("Original list:")
    print_linked_list(head)
    reversed_head = reverseList(head)
    print("Reversed list:")
    print_linked_list(reversed_head)

    l1 = build_linked_list([1, 2, 5, 9])
    l2 = build_linked_list([2, 3, 4, 7, 20])
    print(f"Two init lls:")
    print_linked_list(l1)
    print_linked_list(l2)
    print(f"Merged lls:")
    print_linked_list(mergeTwoLists(l1, l2))

    # binary search
    nums = [-1,0,3,5,9,12]
    target = 9
    print(f"Binary search: {nums} target: {target}, result: {binary_search(nums, target)}, correct 4")

    # rotated array 
    nums = [4,5,6,7,0,1,2]
    target = 0
    print(f"Rotated array search: {nums} target: {target}, result: {search_rotated_array(nums, target)}, correct 4")