

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
    Queue
    '''
    print("\n====== Queue ======")

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

    print("====== Queue ======")
    return



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
