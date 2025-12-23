
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

# ========= Sliding Window =========
def frequency_map(values: List[any]) -> dict:
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



    pass
# ========= Sliding Window =========

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
# ========= Binary Search =========


# ========= BFS =========
def numIslands(grid: List[List[str]]) -> int:
    # https://leetcode.com/problems/number-of-islands/
    # Also same problem https://leetcode.com/problems/max-area-of-island/
    # Assuming we can modify the graph, we can use it to track visited tiles
    pass

# ========= BFS =========


# ========= DFS =========

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