## Leetcode 热门100 (背)

### 148. 排序链表

将链表排序

#### 方法1(Ologn) 自顶向下（快慢指针+递归）

```c++
ListNode* sortList(ListNode* head) {
        return sortList(head, nullptr);
    }

    ListNode* sortList(ListNode* head, ListNode* tail) {
        if (head == nullptr) {
            return head;
        }
        if (head->next == tail) {
            head->next = nullptr;
            return head;
        }
        ListNode* slow = head, *fast = head;
        while (fast != tail) {
            slow = slow->next;
            fast = fast->next;
            if (fast != tail) {
                fast = fast->next;
            }
        }
        ListNode* mid = slow;
        return merge(sortList(head, mid), sortList(mid, tail));
    }

    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode* dummyHead = new ListNode(0);
        ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
        while (temp1 != nullptr && temp2 != nullptr) {
            if (temp1->val <= temp2->val) {
                temp->next = temp1;
                temp1 = temp1->next;
            } else {
                temp->next = temp2;
                temp2 = temp2->next;
            }
            temp = temp->next;
        }
        if (temp1 != nullptr) {
            temp->next = temp1;
        } else if (temp2 != nullptr) {
            temp->next = temp2;
        }
        return dummyHead->next;
    }
```

#### 方法2(O1) 先算长度len在从1到len/2

```c++
ListNode* sortList(ListNode* head) {
        if (head == nullptr) {
            return head;
        }
        int length = 0;
        ListNode* node = head;
        while (node != nullptr) {
            length++;
            node = node->next;
        }
        ListNode* dummyHead = new ListNode(0, head);
        for (int subLength = 1; subLength < length; subLength <<= 1) {
            ListNode* prev = dummyHead, *curr = dummyHead->next;
            while (curr != nullptr) {
                ListNode* head1 = curr;
                for (int i = 1; i < subLength && curr->next != nullptr; i++) {
                    curr = curr->next;
                }
                ListNode* head2 = curr->next;
                curr->next = nullptr;
                curr = head2;
                for (int i = 1; i < subLength && curr != nullptr && curr->next != nullptr; i++) {
                    curr = curr->next;
                }
                ListNode* next = nullptr;
                if (curr != nullptr) {
                    next = curr->next;
                    curr->next = nullptr;
                }
                ListNode* merged = merge(head1, head2);
                prev->next = merged;
                while (prev->next != nullptr) {
                    prev = prev->next;
                }
                curr = next;
            }
        }
        return dummyHead->next;
    }

    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode* dummyHead = new ListNode(0);
        ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
        while (temp1 != nullptr && temp2 != nullptr) {
            if (temp1->val <= temp2->val) {
                temp->next = temp1;
                temp1 = temp1->next;
            } else {
                temp->next = temp2;
                temp2 = temp2->next;
            }
            temp = temp->next;
        }
        if (temp1 != nullptr) {
            temp->next = temp1;
        } else if (temp2 != nullptr) {
            temp->next = temp2;
        }
        return dummyHead->next;
    }
```

---

### 152. 乘积最大子数组 (动态规划，最大最大，最小最小)

重点：子数组即连续，乘积最大

```c++
int maxProduct(vector<int>& nums) {
        vector <int> maxF(nums), minF(nums);
        for (int i = 1; i < nums.size(); ++i) {
            maxF[i] = max(maxF[i - 1] * nums[i], max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = min(minF[i - 1] * nums[i], min(nums[i], maxF[i - 1] * nums[i]));
        }
        return *max_element(maxF.begin(), maxF.end());
    }
```

---

### 155. 最小栈

getMin O1

```c++
class MinStack {
    stack<int> x_stack;
    stack<int> min_stack;
public:
    MinStack() {
        min_stack.push(INT_MAX);
    }
  
    void push(int x) {
        x_stack.push(x);
        min_stack.push(min(min_stack.top(), x));
    }
  
    void pop() {
        x_stack.pop();
        min_stack.pop();
    }
  
    int top() {
        return x_stack.top();
    }
  
    int getMin() {
        return min_stack.top();
    }
};
```

---

### 581. 最短无序连续子数组

给你一个整数数组 `nums` ，你需要找出一个 **连续子数组** ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

请你找出符合题意的 **最短** 子数组，并输出它的长度。

```c++
in:nums = [2,6,4,8,10,9,15]
out:5

int findUnsortedSubarray(vector<int>& nums) {
        int start = nums.size()-1;
        int end = 0;
        int max = 0;
        int min = nums.size()-1;
        for(int i=0;i<nums.size();i++){
            if(nums[max]>nums[i]){
                end = i;
            }else max = i;
        }
        for(int i = nums.size()-1;i>=0;i--){
            if(nums[min]<nums[i]){
                start = i;
            }else min = i;
        }
        return end>start?end-start+1:0;
    }
```

---

### 三数之和

当我们需要枚举数组中的两个元素时，如果我们**发现随着第一个元素的递增，第二个元素是递减的，那么就可以使用双指针的方法**，将枚举的时间复杂度从 O(N2)O(N^2)O(N2) 减少至 O(N)O(N)O(N)。

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};
```

---

### 课程表

拓扑排序

```C++
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> indeg;

public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        indeg.resize(numCourses);
        for (const auto& info: prerequisites) {
            edges[info[1]].push_back(info[0]);
            ++indeg[info[0]];
        }

        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                q.push(i);
            }
        }

        int visited = 0;
        while (!q.empty()) {
            ++visited;
            int u = q.front();
            q.pop();
            for (int v: edges[u]) {
                --indeg[v];
                if (indeg[v] == 0) {
                    q.push(v);
                }
            }
        }

        return visited == numCourses;
    }
};
```

---

### 128. 最长连续子序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现**时间复杂度为 `O(n)` **的算法解决此问题。

<pre><strong>输入：</strong>nums = [100,4,200,1,3,2]
<strong>输出：</strong>4</pre>

```C++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> set(nums.begin(),nums.end());
        int pre;
        int next;
        int ans = 0;
        for(int i=0;i<nums.size();i++){
            if(set.find(nums[i])==set.end())continue;
            pre = nums[i]-1;
            next = nums[i]+1;
            set.erase(nums[i]); 
            while(set.find(pre) != set.end()){
                set.erase(pre--);
            }
            while(set.find(next) != set.end()){
                set.erase(next++);
            }
            ans = max(ans,next-pre-1);
        }
        return ans;
    }
};
```

---

### 238. 除自身以外数组的乘积

给你一个整数数组 `nums`，返回 *数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积* 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在  **32 位** 整数范围内。

请 **不要使用除法，** 且在 `O(<em>n</em>)` 时间复杂度内完成此题。

<pre><strong>输入:</strong> nums = <code>[1,2,3,4]</code>
<strong>输出:</strong> <code>[24,12,8,6]</code></pre>

```C++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ans(nums.size());
        ans[0] = 1;
        int right = 1;
	//先计算i左侧的前缀积，并保存
        for(int i = 1;i<nums.size();i++){
            ans[i] = ans[i-1]*nums[i-1];
        } 
        for(int i = nums.size()-1;i>0;i--){
	    //ans[i] 依赖 i 的左侧前缀积，并且只使用一次，right保存右侧前缀积。
            ans[i] = ans[i] * right;
            right = right * nums[i];
        }
        ans[0] = right;
        return ans;
    }
};
```

---

### 438. 找到字符串中所有字母异位词（滑动窗口+差值记录异同）

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词**的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词**指由相同字母重排列形成的字符串（包括相同的字符串）。

<pre><strong>输入: </strong>s = "cbaebabacd", p = "abc"
<strong>输出: </strong>[0,6]
<strong>解释:</strong>
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。</pre>

```C++
 class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int sLen = s.size(), pLen = p.size();

        if (sLen < pLen) {
            return vector<int>();
        }

        vector<int> ans;
        vector<int> count(26);
        for (int i = 0; i < pLen; ++i) {
            ++count[s[i] - 'a'];
            --count[p[i] - 'a'];
        }

        int differ = 0;
        for (int j = 0; j < 26; ++j) {
            if (count[j] != 0) {
                ++differ;
            }
        }

        if (differ == 0) {
            ans.emplace_back(0);
        }

        for (int i = 0; i < sLen - pLen; ++i) {
            if (count[s[i] - 'a'] == 1) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从不同变得相同（cnt 1 -> 0）
                --differ;
            } else if (count[s[i] - 'a'] == 0) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从相同变得不同 （cnt 0 -> -1）
                ++differ;
            }
            --count[s[i] - 'a'];

            if (count[s[i + pLen] - 'a'] == -1) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从不同变得相同（cnt -1 -> 0）
                --differ;
            } else if (count[s[i + pLen] - 'a'] == 0) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从相同变得不同（cnt 0 -> 1）
                ++differ;
            }
            ++count[s[i + pLen] - 'a'];
  
            if (differ == 0) {
                ans.emplace_back(i + 1);
            }
        }

        return ans;
    }
};
```

---

### 287. 寻找重复数

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

**思路：i ——>nums[i]，当存在重复，必然有 i,j...k ——> nums[i]; 出现环，快慢指针。**

**nums[i] => [1,n]**

**示例 1：**

<pre><strong>输入：</strong>nums = [1,3,4,2,2]
<strong>输出：</strong>2</pre>

```C++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```

---

### 32. 最长有效括号 (双向遍历/栈)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

<pre><strong>输入：</strong>s = "(()"
<strong>输出：</strong>2
<strong>解释：</strong>最长有效括号子串是 "()"</pre>

```

```

---

### 124. 二叉树中的最大路径和

**路径** 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 **至多出现一次** 。该路径** 至少包含一个 **节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

<pre><strong>输入：</strong>root = [1,2,3]
<strong>输出：</strong>6
<strong>解释：</strong>最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6</pre>

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    int ans = INT_MIN;

    int getMax(TreeNode* root){
        if(!root) return 0;
        int l = getMax(root->left);
        int r = getMax(root->right);
        int chose = max(root->val,max(root->val+l,root->val+r));
        ans = max(ans,max(chose,root->val+l+r));
        return chose;
    }

    int maxPathSum(TreeNode* root) {
        getMax(root);
        return ans;
    }
};
```

---

### 239. 滑动窗口最大值 (deque 双端队列，单调队列)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回  *滑动窗口中的最大值 * 。

**示例 1：**

<pre><b>输入：</b>nums = [1,3,-1,-3,5,3,6,7], k = 3
<b>输出：</b>[3,3,5,5,6,7]
<b>解释：</b>
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       <strong>3</strong>
 1 [3  -1  -3] 5  3  6  7       <strong>3</strong>
 1  3 [-1  -3  5] 3  6  7      <strong> 5</strong>
 1  3  -1 [-3  5  3] 6  7       <strong>5</strong>
 1  3  -1  -3 [5  3  6] 7       <strong>6</strong>
 1  3  -1  -3  5 [3  6  7]      <strong>7</strong></pre>

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> q;
        for (int i = 0; i < k; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
        }

        vector<int> ans = {nums[q.front()]};
        for (int i = k; i < n; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
            while (q.front() <= i - k) {
                q.pop_front();
            }
            ans.push_back(nums[q.front()]);
        }
        return ans;
    }
};
```

---

### 395. 至少有 K 个重复字符的最长子串

给你一个字符串 `s` 和一个整数 `k` ，请你找出 `s` 中的最长子串， 要求该子串中的每一字符出现次数都不少于 `k` 。返回这一子串的长度。

**示例 1：**

<pre><strong>输入：</strong>s = "aaabb", k = 3
<strong>输出：</strong>3
<strong>解释：</strong>最长子串为 "aaa" ，其中 'a' 重复了 3 次。
</pre>

**示例 2：**

<pre><strong>输入：</strong>s = "ababbc", k = 2
<strong>输出：</strong>5
<strong>解释：</strong>最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。</pre>

#### 分治

显然，当s[i] 不满足 k的要求，那么只有去除它的子串可能成为结果，这些子串成为分治。

```cpp
class Solution {
public:
    int dfs(const string& s, int l, int r, int k) {
        vector<int> cnt(26, 0);
        for (int i = l; i <= r; i++) {
            cnt[s[i] - 'a']++;
        }

        char split = 0;
        for (int i = 0; i < 26; i++) {
            if (cnt[i] > 0 && cnt[i] < k) {
                split = i + 'a';
                break;
            }
        }
        if (split == 0) {
            return r - l + 1;
        }

        int i = l;
        int ret = 0;
        while (i <= r) {
            while (i <= r && s[i] == split) {
                i++;
            }
            if (i > r) {
                break;
            }
            int start = i;
            while (i <= r && s[i] != split) {
                i++;
            }

            int length = dfs(s, start, i - 1, k);
            ret = max(ret, length);
        }
        return ret;
    }

    int longestSubstring(string s, int k) {
        int n = s.length();
        return dfs(s, 0, n - 1, k);
    }
};
```

#### 滑动窗口

利用右移增加字符左移减少字符，我们规定1-26种字符出现的结果并遍历，当窗口中的字符种类分别等于1-26并且当前窗口满足k条件则它是一个可能解。

```cpp
class Solution {
public:
    int longestSubstring(string s, int k) {
        int ret = 0;
        int n = s.length();
        for (int t = 1; t <= 26; t++) {
            int l = 0, r = 0;
            vector<int> cnt(26, 0);
            int tot = 0;
            int less = 0;
            while (r < n) {
                cnt[s[r] - 'a']++;
                if (cnt[s[r] - 'a'] == 1) {
                    tot++;
                    less++;
                }
                if (cnt[s[r] - 'a'] == k) {
                    less--;
                }

                while (tot > t) {
                    cnt[s[l] - 'a']--;
                    if (cnt[s[l] - 'a'] == k - 1) {
                        less++;
                    }
                    if (cnt[s[l] - 'a'] == 0) {
                        tot--;
                        less--;
                    }
                    l++;
                }
                if (less == 0) {
                    ret = max(ret, r - l + 1);
                }
                r++;
            }
        }
        return ret;
    }
};
```

---

### 有序矩阵中的第K小的元素

给你一个 `n x n` 矩阵 `matrix` ，其中每行和每列元素均按升序排序，找到矩阵中第 `k` 小的元素。
请注意，它是 **排序后** 的第 `k` 小元素，而不是第 `k` 个 **不同** 的元素。

你必须找到一个内存复杂度优于 `O(n<sup>2</sup>)` 的解决方案。

**示例 1：**

<pre><strong>输入：</strong>matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
<strong>输出：</strong>13
<strong>解释：</strong>矩阵中的元素为 [1,5,9,10,11,12,13,<strong>13</strong>,15]，第 8 小元素是 13
</pre>

**示例 2：**

<pre><strong>输入：</strong>matrix = [[-5]], k = 1
<strong>输出：</strong>-5</pre>

```cpp
class Solution {
public:
    bool check(vector<vector<int>>& matrix, int mid, int k, int n) {
        int i = n - 1;
        int j = 0;
        int num = 0;
        while (i >= 0 && j < n) {
            if (matrix[i][j] <= mid) {
                num += i + 1;
                j++;
            } else {
                i--;
            }
        }
        return num >= k;
    }

    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size();
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (check(matrix, mid, k, n)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
};

作者：力扣官方题解
链接：https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/solutions/311472/you-xu-ju-zhen-zhong-di-kxiao-de-yuan-su-by-leetco/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

---

### 多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1：**

<pre><strong>输入：</strong>nums = [3,2,3]
<strong>输出：</strong>3</pre>

**示例 2：**

<pre><strong>输入：</strong>nums = [2,2,1,1,1,2,2]
<strong>输出：</strong>2</pre>

```cpp
int majorityElement(vector<int>& nums) {
        int more = nums[0];// 当前的擂主
        int cur = 0;
        for(int n:nums){
            if(more==n)cur++;//同一阵营，壮大
            else cur--;//不同削弱
            if(cur==0){//阵营为0，我为擂主
                cur++;
                more = n;
            }
        }
        return more;//只有阵营人数最多的会成为擂主
    }
```

---

### 448. 找到所有数组中消失的数字

给你一个含 `n` 个整数的数组 `nums` ，其中 `nums[i]` 在区间 `[1, n]` 内。请你找出所有在 `[1, n]` 范围内但没有出现在 `nums` 中的数字，并以数组的形式返回结果。

**示例 1：**

<pre><strong>输入：</strong>nums = [4,3,2,7,8,2,3,1]
<strong>输出：</strong>[5,6]
</pre>

**示例 2：**

<pre><strong>输入：</strong>nums = [1,1]
<strong>输出：</strong>[2]</pre>

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        for (auto& num : nums) {
            int x = (num - 1) % n;
            nums[x] += n; // 对于遇到的数，所在下标增加n,这使得 遇到的每个数的下标对应都大于n，如果不在数组里的数，其对应下标不变，则永远小于等于n。
        }
        vector<int> ret;
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                ret.push_back(i + 1);
            }
        }
        return ret;
    }
};
```

---

### 剑指 Offer II 005. 单词长度的最大乘积

给定一个字符串数组 `words`，请计算当两个字符串 `words[i]` 和 `words[j]` 不包含相同字符时，它们长度的乘积的最大值。假设字符串中只包含英语的小写字母。如果没有不包含相同字符的一对字符串，返回 0。

<pre><strong>输入:</strong> words = <code>["abcw","baz","foo","bar","fxyz","abcdef"]</code>
<strong>输出: </strong><code>16 
<strong>解释:</strong> 这两个单词为<strong></strong></code><code>"abcw", "fxyz"</code>。它们不包含相同字符，且长度的乘积最大。</pre>

```cpp
class Solution {
public:
    int maxProduct(vector<string>& words) {
        int length = words.size();
        vector<int> masks(length);
	// 计算每一个word的位，例如abc（0，1，2）等价于 111. 当出现其它111时表示相等。 关键点在于meet 与 met 的 位掩码一致。并且 26 个字母 int 可以容纳。
        for (int i = 0; i < length; i++) {
            string word = words[i];
            int wordLength = word.size();
            for (int j = 0; j < wordLength; j++) {
                masks[i] |= 1 << (word[j] - 'a');
            }
        }
        int maxProd = 0;
	//遍历words数组获得乘积最大的字符串
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                if ((masks[i] & masks[j]) == 0) {
                    maxProd = max(maxProd, int(words[i].size() * words[j].size()));
                }
            }
        }
        return maxProd;
    }
};
```

---

### 剑指 Offer II 004. 只出现一次的数字

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现  **三次 。** 请你找出并返回那个只出现了一次的元素。

<pre><strong>输入：</strong>nums = [2,2,3,2]
<strong>输出：</strong>3</pre>

```cpp
/*
对于类似的多个相同数消除 ，可以考虑状态转换，并列出状态转换表
例如2个相同数可以使用异或消除
对于三个数，我们考虑状态转换为，00-》01-》10-》00，
所以有真值表
a b x a b（new）
0 0 0 0 0
0 1 0 0 1
1 0 0 1 0
0 0 1 0 1
0 1 1 1 0
1 0 1 0 0
*/

考虑，计算 a = ab'x' + a'bx, b = a'bx' +  a'b'x
更进一步 先计算b 然后带入新b计算新 a 如下代码：

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int a = 0, b = 0;
        for (int num: nums) {
            b = ~a & (b ^ num);
            a = ~b & (a ^ num);
        }
        return b;
    }
};
```

---

### 剑指 Offer II 010. 和为 k 的子数组（有负数，滑动窗口划不动，前缀和）

给定一个整数数组和一个整数 `k` ** ，** 请找到该数组中和为 `k` 的连续子数组的个数。

<pre><strong>输入:</strong>nums = [1,1,1], k = 2
<strong>输出:</strong> 2
<strong>解释:</strong> 此题 [1,1] 与 [1,1] 为两种不同的情况</pre>

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1;
        int count = 0, pre = 0;
        for (auto& x:nums) {
            pre += x;
            if (mp.find(pre - k) != mp.end()) {
                count += mp[pre - k];
            }
            mp[pre]++;
        }
        return count;
    }
};
```

---

### 剑指 Offer II 058. 日程表 (知识点：线段树、树状数组)

请实现一个 `MyCalendar` 类来存放你的日程安排。如果要添加的时间内没有其他安排，则可以存储这个新的日程安排。

`MyCalendar` 有一个 `book(int start, int end)`方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 `[start, end)`, 实数 `x` 的范围为，  `start <= x < end`。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生重复预订。

每次调用 `MyCalendar.book`方法时，如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 `true`。否则，返回 `false` 并且不要将该日程安排添加到日历中。

请按照以下步骤调用 `MyCalendar` 类: `MyCalendar cal = new MyCalendar();` `MyCalendar.book(start, end)`


---

### 查找和最小的 K 对数字 （知识点：多路归并）

给定两个以 **升序排列** 的整数数组 `nums1` 和 `nums2` , 以及一个整数 `k` 。

定义一对值 `(u,v)`，其中第一个元素来自 `nums1`，第二个元素来自 `nums2` 。

请找到和最小的 `k` 个数对 `(u<sub>1</sub>,v<sub>1</sub>)`, ` (u<sub>2</sub>,v<sub>2</sub>)`  ...  `(u<sub>k</sub>,v<sub>k</sub>)` 。

思路：相当于寻找两路中较小的值，由于数组已经排序，那么我们知道，下标0，0对肯定是最小值，只需要比较0，1和1，0对那个和小，因此我们提前放入1，0 --n，0，然后依次放入0，1---0，n，单次放入时弹出较大的（1，0或者0，1），剩下的必然是较小值，在执行k次后得到结果。

[(nums1[0], nums2[0]), (nums1[0], nums2[1]), ..., (nums1[0], nums2[m - 1])]

[(nums1[1], nums2[0]), (nums1[1], nums2[1]), ..., (nums1[1], nums2[m - 1])]

[(nums1[n - 1], nums2[0]), (nums1[n - 1], nums2[1]), ..., (nums1[n - 1], nums2[m - 1])]

---

### 剑指 Offer II 067. 最大的异或

给你一个整数数组 `nums` ，返回 `nums[i] XOR nums[j]` 的最大运算结果，其中 `0 ≤ i ≤ j < n` 。

**x = ai xor aj =》ai = x xor aj，我们希望x越大越好，所以应该从x的高位开始判断是否满足该位能否为1，因此我们依次遍历31位，从高位开始，并保留高位的结果，在高位最佳的前提下探索下一位的最优结果。所以我们可以保存nums所有高n位的情况到hash表中，并再次遍历nums[j] xor x_n位。如果存在则高位为1，否则为0。为1则x = x***2 +1，为0则为x -1（x已经先假设为1了）。

**示例 1：**

<pre><strong>输入：</strong>nums = [3,10,5,25,2,8]
<strong>输出：</strong>28
<strong>解释：</strong>最大运算结果是 5 XOR 25 = 28.</pre>

```cpp
class Solution {
private:
    // 最高位的二进制位编号为 30
    static constexpr int HIGH_BIT = 30;

public:
    int findMaximumXOR(vector<int>& nums) {
        int x = 0;
        for (int k = HIGH_BIT; k >= 0; --k) {
            unordered_set<int> seen;
            // 将所有的 pre^k(a_j) 放入哈希表中
            for (int num: nums) {
                // 如果只想保留从最高位开始到第 k 个二进制位为止的部分
                // 只需将其右移 k 位
                seen.insert(num >> k);
            }

            // 目前 x 包含从最高位开始到第 k+1 个二进制位为止的部分
            // 我们将 x 的第 k 个二进制位置为 1，即为 x = x*2+1
            int x_next = x * 2 + 1;
            bool found = false;
      
            // 枚举 i
            for (int num: nums) {
                if (seen.count(x_next ^ (num >> k))) {
                    found = true;
                    break;
                }
            }

            if (found) {
                x = x_next;
            }
            else {
                // 如果没有找到满足等式的 a_i 和 a_j，那么 x 的第 k 个二进制位只能为 0
                // 即为 x = x*2
                x = x_next - 1;
            }
        }
        return x;
    }
};
```

---

###
