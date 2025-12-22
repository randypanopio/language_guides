
import heapq
from collections import deque


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

if __name__ == "__main__":
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