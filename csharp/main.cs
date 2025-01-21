public class DataStructures {

    //System.Collections.Generic Namespace
    // https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic?view=net-9.0
    
    #region Dictionary
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.dictionary-2?view=net-9.0
    // Declaration
    var dict = new Dictionary<T, T>();
    // Inline Declaration
    var dict = new Dictioanry<T,T>() {
        {"foo", "bar"}, {"key", "value"} 
    };
    // Add (if key does not exist)
    dict.Add("key", "value");
    // Change Key's Value AND assign value if key does not yet exist
    dict["key"] = "newValue";
    // TryGetValue - used to check if keys exist and immediately get the value
    dict.TryGetValue("key", out T value);
    // Contains Key
    bool dictContains = dict.ContainsKey("key");
    // Contains Value, likely O(n)
    bool dictValContains = dict.ContainsValue("value");   
    #endregion

    # region HashSet
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.hashset-1?view=net-9.0
    // Declaration
    var hashSet = new HashSet<T>();
    // Add element
    hashSet.Add("value");
    // Remove element
    hashSet.Remove("value");
    // Contains element
    bool contains = hashSet.Contains("value");
    // UnionWith another HashSet
    hashSet.UnionWith(new HashSet<T> {"value1", "value2"});
    // IntersectWith another HashSet
    hashSet.IntersectWith(new HashSet<T> {"value2", "value3"});
    // Clear HashSet
    hashSet.Clear();
    #endregion

    # region Stack
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.stack-1?view=net-9.0
    // Declaration
    var stack = new Stack<T>();
    // Push element
    stack.Push("value");
    // Pop element
    var top = stack.Pop();
    // Peek element
    var peek = stack.Peek();
    // Check if empty, and also how many elements in stack
    bool isEmpty = stack.Count == 0;
    #endregion

    # region Queue
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.queue-1?view=net-9.0
    // Declaration
    var queue = new Queue<T>();
    // Enqueue element
    queue.Enqueue("value");
    // Dequeue element
    var front = queue.Dequeue();
    // Peek element
    var peekQueue = queue.Peek();
    // Check if empty, and also how many elements
    bool isEmptyQueue = queue.Count == 0;
    #endregion

    # region PriorityQueue
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.priorityqueue-2?view=net-9.0
    // Declaration
    var priorityQueue = new PriorityQueue<TElement, TPriority>();
    // Enqueue element with priority
    priorityQueue.Enqueue("element", priority);
    // Dequeue element with highest priority
    var dequeued = priorityQueue.Dequeue();
    // Peek element with highest priority
    var peekPriority = priorityQueue.Peek();
    // Check if empty
    bool isEmptyPriority = priorityQueue.Count == 0;
    #endregion

    # region LinkedList
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.linkedlist-1?view=net-9.0
    // Declaration
    var linkedList = new LinkedList<T>();
    // Add to the front
    linkedList.AddFirst("value");
    // Add to the back
    linkedList.AddLast("value");
    // Remove element
    linkedList.Remove("value");
    // Access first element
    var first = linkedList.First;
    // Access last element
    var last = linkedList.Last;
    #endregion

    # region List
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.list-1?view=net-9.0
    // Declaration
    var list = new List<T>();
    // Add element
    list.Add("value");
    // Remove element
    list.Remove("value");
    // Access element by index
    var element = list[0];
    // Insert at index
    list.Insert(0, "value");
    // Remove at index
    list.RemoveAt(0);
    // Sort list
    list.Sort();
    #endregion

    # region Arrays
    // Documentation: https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/arrays/
    // Declaration
    var array = new T[10];
    // Access element by index
    var elementArray = array[0];
    // Set element by index
    array[0] = "value";
    // Length of array
    var length = array.Length;
    // Iterate over array
    foreach (var item in array) {
        Console.WriteLine(item);
    }
    #endregion

    # region String
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.string?view=net-9.0
    // Declaration
    string str = "example";
    // Concatenation
    string concat = str + " more text";
    // Substring
    string substr = str.Substring(0, 3);
    // Contains
    bool containsStr = str.Contains("exam");
    // Split
    string[] parts = str.Split(' ');
    // Replace
    string replaced = str.Replace("exam", "test");
    // Length
    int strLength = str.Length;
    // Iterate over characters
    foreach (var ch in str) {
        Console.WriteLine(ch);
    }
    #endregion

    # region KeyValuePairs
    // Documentation: https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.keyvaluepair-2?view=net-9.0
    // KeyValuePair Declaration
    var kvp = new KeyValuePair<T, T>("key", "value");
    // Access key
    var key = kvp.Key;
    // Access value
    var value = kvp.Value;
    #endregion
}

class BasicAlgorithms {
    #region Custom Sorting - OrderBy (for an IEnumerable)
    // OrderBy, pass a delegate for a custom comparison
    var toCompare = new List<int> { 5, 2, 8, 3, 9 };
    var sorted = toCompare.OrderBy((a,b) => Compare(a,b));
    var revSorted = toCompare.OrderByDescending((a,b) => Compare(a,b));

    static int Compare (T a, T b) {
        int ia = (int)a;
        int ib = (int)b;
        return b.CompareTo(b); // custom logic here
    }
    #endregion

    #region Frequency Map
    // Used to generate a hashtable/dict where each number represented as key, dictates the amount of occurance in the array.
    static Dictionary<int, int> GenerateFrequencyMap(int[] numbers) {
        var frequencyMap = new Dictionary<int, int>();
        foreach (int number in numbers) {
            if (frequencyMap.ContainsKey(number)) {
                frequencyMap[number]++;
            } else {
                frequencyMap[number] = 1;
            }
        }
        return frequencyMap;
    }
    
    public int[] TopKFrequent(int[] nums, int k) {
        // https://leetcode.com/problems/top-k-frequent-elements/
        // Step 1 generate freq map O(n)
        Dictionary<int, int> freq = new Dictionary<int, int>();
        foreach (int num in nums) {
            if (freq.ContainsKey(num)){
                freq[num]++;
            } else {
                freq[num]=1;
            }
        }

        // Step 2 generate buckets for freqencies O(n)
        // a bucket will contain a list of numbers with the same frequency will be grouped together
        // we create from frequency 0 to maximum frequency, and add the numbers into the frequency bucket according to index
        // eg at index i, it will represent numbers that occur exactly i times (index 0 in this case will have no occurance)
        List<int>[] buckets = new List<int>[nums.Length+1];
        foreach (int key in freq.Keys){
            int occurance = freq[key];
            if (buckets[occurance] == null) {
                buckets[occurance] = new List<int>();
            }
            buckets[occurance].Add(key); 
        }
        
        // increment count using index as the number to increment
        // step 3 Build the resulting array of up to K length
        // Iterate backwards from len (max occurance) until we reach k elements
        int[] result = new int[k];
        int index = 0; // used to track where to insert in our resulting array, this lets us skip casting the list into an array 
        for (int i = buckets.Length - 1; i >= 0; i--) {
            if (buckets[i] != null) { // this bucket has no nums of this frequency
                foreach (int item in buckets[i]) {
                    result[index] = item; // add all nums in this bucket to our resulting answer
                    index++; // increment after weve added our answer
                    if (index == k) {
                        return result; // Stop early when we have k elements.
                    }
                }
            }
        }

        return result;
    }    
    #endregion 

    #region Two pointers
    /*
        Logic, use two pointers from left to right to find something
        for example anagram, or 2sum
    */
    ///
    public bool IsPalindrome(string s) {
        // https://leetcode.com/problems/valid-palindrome/
        var left = 0;
        var right = s.Length - 1;
        while (left < right){
            // check for non alphanumerics
            while(left < right && !Char.IsLetterOrDigit(s[left])) {
                left++;
            }
            while(left < right && !Char.IsLetterOrDigit(s[right])) {
                right--;
            }            
            if (Char.ToLower(s[left]) != Char.ToLower(s[right])) {
                return false; // found a mismatch
            }
            // move both ptrs inwards, currently valid palindrome so far
            left++;
            right--;
        }
        return true; // reached the crossing, no mismatched so is valid palindrome
    }

    // twosum 2
    public int[] TwoSum(int[] numbers, int target) {
        // https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
        // condition, the array is sorted, so we are finding numbers that add up to target. Target = L + R
        var left = 0; 
        var right = numbers.Length - 1;
        while (left < right){
            var current = numbers[left] + numbers[right]; // attempt to calculate target and check
            if (current == target){ // we found Target = L + R, return the indices
                return [left + 1, right + 1]; // condition of the question, we add + 1 from a 0 based index
            }
            if (current > target) {
                right--; // start decrementing
            } else {
                left++; // reached negative, start moving left
            }
        }
        return [-1,-1]; // reached middle without finding a target
    }

    // 3sum
    public IList<IList<int>> ThreeSum(int[] nums) {
        // https://leetcode.com/problems/3sum/
        // basically we rearrange the solutions so that its a = -(b + c) instead of a + b + c = 0
        IList<IList<int>> result = new List<IList<int>>();
        Array.Sort(nums); // sort the array so we can take advantage of using 2sum2 logic, this also enables reduction of duplicates
        for (int i = 0; i < nums.Length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]){ continue; } // Skip duplicate first numbers
            // 2 sum II solution approach, find the 2 sum sliding window since the array is sorted and skipping duplicates
            int target = -nums[i]; // instead of a + b + c = 0, we solve for -a = b + c where -a is the target
            int left = i + 1; // since the array is sorted, the numbers that sum up to target must be to the right of i
            int right = nums.Length - 1;
            while (left < right) { // core finding target a
                int sum = nums[left] + nums[right];
                if (sum == target) {
                    // Add the triplet to the result
                    result.Add(new List<int> { nums[i], nums[left], nums[right] });

                    // now we need to push the pointers of L/R so we skip duplicates
                    // we simply check if the neighboring numbers are the same numbers, we push our indices in that case, doing so skips duplicates
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++; // Move both pointers after processing this triplet
                    right--; // we continue the left/right to check if there are additional triplets we could use
                } else if (sum < target) {
                    left++; // Move left pointer to increase the sum
                } else {
                    right--; // Move right pointer to decrease the sum
                }
            }
        }
        return result; // another way would have been a triple loop brute force
    #endregion

    #region Sliding Window
    public int MaxProfit(int[] prices) {
        // https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
        // not actually the most optimal solution for this, but its a basis for sliding window
        int left = 0, right = 1, max = 0;
        while (right < prices.Length) {
            if (prices[right] > prices[left]) { // we are in an up swing, we check 
                int profit = prices[right] - prices[left]; // calculate the profit
                max = Math.Max(max, profit); // compare to current max profit
            } else {
                left = right; // move left pointer to a downswing
            }
            right++; // always increment right pointer
        }
        return max;
    }

    #endregion

    #region Kadane's algorithm - Maximum Sub Array
    // TODO
    #endregion

    #region Binary Search
    public int Search(int[] nums, int target) {
        // https://leetcode.com/problems/binary-search/
        int left = 0, right = nums.Length -1;
        while(left <= right) {
            int mid = left + ((right - left) / 2); // get midpoint without overflow
            if (nums[mid] == target) {
                return mid; // found
            } else if (nums[mid] < target) { // target space is in the right half
                left = mid + 1;
            } else { // otherwise it is in the left half
                right = mid - 1;
            }
        } 
        return -1; // not found
    }
    #endregion

    #region Reverse Iteration
    public void countBackwards(IEnumerable<int> nums)
        for (int i = nums.Length - 1; i >= 0; i--) {
            var foo = nums[i];
        }
    #endregion


    #region Linked List
 public class ListNode {
     public int val;
     public ListNode next;
     public ListNode(int val=0, ListNode next=null) {
         this.val = val;
         this.next = next;
     }
 }

    public ListNode ReverseList(ListNode head) {
        // https://leetcode.com/problems/reverse-linked-list/
        // by iteratively pointing to the next node while updating each node's next to point to it's previous, the LL reverses.
        ListNode prev = null; // Initialize prev as null, as the new tail of the list will point to null
        while (head != null) {
            var next = head.next; // Save a reference to the next node
            head.next = prev; // Reverse the 'next' pointer of the current node to point to the previous node
            prev = head; // Move prev to the current node for the next iteration
            head = next; // Move to the next node in the list
        }
        return prev; // After the loop, prev will be the new head of the reversed list
    }

    public ListNode MergeTwoLists(ListNode list1, ListNode list2) {
        // https://leetcode.com/problems/merge-two-sorted-lists/
        if (list1 == null) { return list2; } // base cases
        if (list2 == null) { return list1; } // base cases
        ListNode result = new ListNode(0);
        ListNode current = result; // Pointer to track the current node in the merged list        
        // Traverse both lists and merge them
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;  // Attach the smaller node to current
                list1 = list1.next;    // Move list1 forward
            } else {
                current.next = list2;  // Attach the smaller node to current
                list2 = list2.next;    // Move list2 forward
            }
            current = current.next; // Move the current pointer forward
        }        
        if (list1 != null) { // If there are remaining nodes in either list, attach them
            current.next = list1;
        } else if (list2 != null) {
            current.next = list2;
        }
        return result.next; // The head of the merged list is the next of the dummy node
    }

    public bool HasCycle(ListNode head) {
        // https://leetcode.com/problems/linked-list-cycle/
        // slow fast pointers, even with using the fast pointer here as the condition, the two will still meet on a cycle, AND its still bound to O(n)
        if (head == null || head.next == null) return false; // base cases
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) { // we use fast pointer as condition since it either loops or slow catches
            slow = slow.next;
            fast = fast.next.next; // we can do this because even if it is null, the slow pointer will catch up, or we looped
            if (slow == fast) { return true; } // cycle detected 
        } 
        return false; // iterated through 
    }
    #endregion

    
}

#region Basic Algorithms






#endregion
