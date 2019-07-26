### 《剑指Offer》专题: 第六期 51-66
大家好，又是好久没更新了，趁有时间把剑指Offer专题收尾工作完成，这里是《剑指Offer》第六期，本期更新的是最后的16题，一起来看看吧~

#### 题目51：构建乘积数组
<center>**juzhong**</center>
给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

```python
"""
引自链接：https://www.nowcoder.com/questionTerminal/94a4d381a68b47b7a8bed86f2975db46
B[i]的值可以看作下图的矩阵中每行的乘积。
下三角用连乘可以很容求得，上三角，从下向上也是连乘。
因此我们的思路就很清晰了，先算下三角中的连乘，即我们先算出B[i]中的一部分，
然后倒过来按上三角中的分布规律，把另一部分也乘进去。
"""
class Solution:
    def multiply(self, A):
        # write code here
        n = len(A)
        if not A or n <=1:return [0]
        B = [1]*n
        for i in range(1,n):
            B[i] = B[i-1]*A[i-1]
        temp = 1
        for j in range(n-2,-1,-1):
            temp *= A[j+1]
            B[j] *= temp
        return B
```
![](https://uploadfiles.nowcoder.com/images/20160829/841505_1472459965615_8640A8F86FB2AB3117629E2456D8C652)
#### 题目52：正则表达式匹配
请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
```python
"""
方法1思路脉络：递归
首先是分析 两个都是空串、其中一个是空串（两种）的情况；然后是分析 两个都是非空串的情况
    
对于非空的两个串，判断模式串的第二位是不是 “*”
    
    如果是“*”，且第一位不等，则递归判断模式串pattern[2:]与字符串S是否match
    如果是“*”，且第一位相等，则分为三种情况 分别递归
        这三种情况举例说明一下：
        s不变，   pattern后移两位： s="aba",pattern="a*aba"
        s后移一位，pattern后移两位： s="aba",pattern="a*ba"
        s后移一位，pattern不变   ： s="aaab",pattern = "a*b"
    
    如果不是“*”，且第一位相等，则递归判断pattern[1:]与s[1:]是否匹配
    如果不是“*”，且第一位不相等，返回False
    
小提示：方法1采用了切片的方式递归，我们还可以用加两个参数的形式递归，不用切片，节省空间
"""
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        #方法2：
        def imatch(s, pattern, s_start, p_start, s_end, p_end):
            if s_end - s_start == 0 and p_end - p_start == 0:
                return True
            elif s_end - s_start != 0 and p_end - p_start == 0:
                return False
            elif s_end - s_start == 0 and p_end - p_start != 0:
                if p_end - p_start+1 >= 2 and pattern[p_start+1] == '*':
                    return imatch(s,pattern,s_start,p_start+1,s_end,p_end)
                else:
                    return False
            else:
                if p_end - p_start+1>=2 and pattern[p_start+1] == '*':
                    if s[s_start] != pattern[p_start] and pattern[p_start] != '.':
                        return imatch(s,pattern,s_start,p_start+2,s_end,p_end)
                    else:
                        return imatch(s,pattern,s_strat,p_start+2,s_end,p_end)\
                                or imatch(s,pattern,s_start+1,p_start+2,s_end,p_end)\
                                or imatch(s,pattern,s_start+1,p_start,s_end,p_end)
                else:
                    if s[s_start]==pattern[p_start] or pattern[p_start] == '.':
                        return imatch(s,pattern,s_start+1,p_start+1,s_end,p_end)
                    else:
                        return False
        #缺少鲁棒性代码记得补上
        imatch(s,pattern,0,0,len(s)-1,len(pattern)-1)
        # 方法1  递归
        """
        # 如果s与pattern都为空，则True
        if len(s) == 0 and len(pattern) == 0:
            return True
        # 如果s不为空，而pattern为空，则False
        elif len(s) != 0 and len(pattern) == 0:
            return False
        # 如果s为空，而pattern不为空，则需要判断
        elif len(s) == 0 and len(pattern) != 0:
            # pattern中的第二个字符为*，则pattern后移两位继续递归比较
            if len(pattern) >= 2 and pattern[1] == '*':
                return self.match(s, pattern[2:])
            else:
                return False
        # s与pattern都不为空的情况
        else:
            # pattern的第二个字符为*的情况
            if len(pattern) >= 2 and pattern[1] == '*':
                # s与pattern的第一个元素不同，则s不变，pattern后移两位，相当于pattern前两位当成空
                if s[0] != pattern[0] and pattern[0] != '.':
                    return self.match(s, pattern[2:])
                else:
                    # 如果s[0]与pattern[0]相同，且pattern[1]为*，这个时候有三种情况
                    # pattern后移2个，s不变；相当于把pattern前两位当成空，匹配后面的
                    # pattern后移2个，s后移1个；相当于pattern前两位与s[0]匹配
                    # pattern不变，s后移1个；相当于pattern前两位，与s中的多位进行匹配，因为*可以匹配多位
                    return self.match(s, pattern[2:]) or self.match(s[1:], pattern[2:]) or self.match(s[1:], pattern)
            # pattern第二个字符不为*的情况
            else:
                if s[0] == pattern[0] or pattern[0] == '.':
                    return self.match(s[1:], pattern[1:])
                else:
                    return False
        """
```
#### 题目53：表示数值的字符串
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
```python
"""
解法：需满足一下几个条件
A：s串非空
B：e或E不能做最后的一位
C:.号不能出现两次，且不能出现在e或E之后
D: 第二个加减+-只能出现在e或E的后面
E: 得是数字和规定符号字符
"""
class Solution:
    def isNumeric(self, s):
        if s == None or s == '\n':return False#最开始的判断
        sign, dot, eE = False, False, False #表示是否出现过
        n = len(s)
        for i in range(n):
            ch = s[i]
            if ch=='e' or ch == 'E': #e或E不能做最后的一位
                if i == n-1:return False
                if eE:return False
                eE = True
            elif ch == '.':# .号不能出现两次，且不能出现在e或E之后
                if dot or eE:return False
                dot = True
            elif ch =='+' or ch == '-': # 第二个加减+-只能出现在E的后面
                if sign and (s[i-1] != 'E' and s[i-1] != 'e'):
                    return False 
                if (sign == False) and (i>0) and (s[i-1] != 'E' and s[i-1] != 'e'):
                    return False
            elif ch <'0' or ch >'9':#非法字符
                return False
        return True
```
#### 题目54：字符流中第一个不重复的字符
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
如果当前字符流没有存在出现一次的字符，返回#字符。
```python
'''
    解法：利用一个int型数组表示256个字符，这个数组初值置为-1.
    每读出一个字符，将该字符的位置存入字符对应数组下标中。
    若值为-1标识第一次读入，不为-1且>0表示不是第一次读入，将值改为-2.
    之后在数组中找到>0的最小值，该数组下标对应的字符为所求。
    在python中，ord(char)是得到char对应的ASCII码；chr(idx)是得到ASCII位idx的字符
'''
class Solution:
    def __init__(self):
        self.char_list = [-1]*256
        self.index = 0  # 记录当前字符的个数，可以理解为输入的字符串中的下标

    def FirstAppearingOnce(self):
        # write code here
        min_value = 0XFFFF
        min_idx = -1
        for i in range(256):
            if self.char_list[i] > -1:
                if self.char_list[i] < min_value:
                    min_value = self.char_list[i]
                    min_idx = i
        if min_idx > -1:
            return chr(min_idx)
        else:
            return '#'
    def Insert(self, char):
        # 如果是第一出现，则将对应元素的值改为下边
        if self.char_list[ord(char)] == -1:
            self.char_list[ord(char)] = self.index
        # 如果已经出现过两次了，则不修改
        elif self.char_list[ord(char)] == -2:
            pass
        # 如果出现过一次，则进行修改，修改为-2
        else:
            self.char_list[ord(char)] = -2
        self.index += 1
```
#### 题目55：链表中环的入口结点
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
```python
"""
快慢指针，慢指针每次走一步，快指针每次走两步，他们总是会在环中相遇；
然后让两个指针一个从链表头走，一个从相遇点走，他们就会想遇到环的入口点
"""
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        try:
            fast = pHead.next
            slow = pHead
            while fast != slow: #若存在循环，快慢指针终会相遇
                fast = fast.next.next
                slow = slow.next
        except:
            # 如果fast != slow，没有循环
            return None

        # 第二步：since fast starts at head.next, we need to move slow one step forward
        slow = slow.next
        while pHead is not slow:#根据注释的分析，head和slow会在进入循环结点出相遇
            pHead = pHead.next
            slow = slow.next
        return pHead
```
#### 题目56：删除链表中重复的结点
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
```python
"""
时间复杂度O（n）,空间复杂度O(n)
思路：例子1233445，先变成12345,再变成125
"""
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead:return pHead
        p = pHead
        temp = set()
        while p and p.next:
            if p.val == p.next.val:
                temp.add(p.val)
                p.next = p.next.next
            else:
                p = p.next
        
        q = ListNode("*")
        q.next= pHead
        x = q
        while x and x.next:
            if x.next.val in temp:
                x.next = x.next.next
            else:
                x = x.next
        return q.next
```
#### 题目57：二叉树的下一个结点
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
```python
"""
1.二叉树为空，则返回空；

2.节点右孩子存在，则设置一个指针从该节点的右孩子出发，
一直沿着指向左子结点的指针找到的叶子节点即为下一个节点；

3.节点不是根节点。如果该节点是其父节点的左孩子，则返回父节点；
否则继续向上遍历其父节点的父节点，重复之前的判断，返回结果。代码如下：
"""
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode: return None
        if pNode.right!=None:
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            return pNode
        while pNode.next != None: #next是父节点
            proof = pNode.next
            if proof.left == pNode:
                return proof
            else:
                pNode = pNode.next
        return None
```
#### 题目58：对称的二叉树
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
```python
"""
可以使用递归的方式：
满足，左子节点等于右子节点的值，且左右节点的子树也是对称的
当然，空节点默认对称。
"""
class Solution:
    def isSymmetrical(self, pRoot):
        def isSym(left,right):
            if (left == None and right == None):
                return True
            if (left == None or right == None): 
                return False
            return left.val == right.val and isSym(left.left,right.right) and isSym(left.right,right.left)
        if not pRoot: return True
        return isSym(pRoot.left,pRoot.right)
```
#### 题目59：按之字形顺序打印二叉树
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
```python
"""
广度优先遍历，遇到偶数行反过来
row_list存在当前层的遍历，child_list存下下一层的遍历。
"""
class Solution:
    def Print(self, pRoot):
        # write code here
        if pRoot == None:return []
        result = []
        queue = [pRoot]
        row = 1
        while queue != []:
            child_list = []
            row_list = []
            for i in queue:#遍历一层存到row_list,然后把下一层存在child_list.
                row_list.append(i.val)
                if i.left != None:
                    child_list.append(i.left)
                if i.right != None:
                    child_list.append(i.right)
            queue = child_list
            if row %2 == 0:#偶数层反过来
                row_list.reverse()
            result.append(row_list)
            row += 1
        return result
```
#### 题目60：把二叉树打印成多行
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
```python
"""
和上一题差不多，不要判断奇数偶数
"""
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if pRoot == None:return []
        result = []
        queue = [pRoot]
        while queue != []:
            child_list = []
            row_list = []
            for i in queue:#遍历一层存到row_list,然后把下一层存在child_list.
                row_list.append(i.val)
                if i.left != None:
                    child_list.append(i.left)
                if i.right != None:
                    child_list.append(i.right)
            queue = child_list
            result.append(row_list)
        return result
```
#### 题目61：序列化二叉树
请实现两个函数，分别用来序列化和反序列化二叉树
```python
"""
思路：递归实现序列化和反序列化，要包括空节点才行，这样可以通过一次遍历的结果复原二叉树。
如果不给空节点做标记，则需要用前中后三种遍历中的两种才可以复原。
"""
class Solution:
    def __init__(self):
        self.flag = -1#用来计数长度
    #递归序列化
    def Serialize(self, root):
        if not root:
            return '#,'
        return str(root.val)+','+self.Serialize(root.left)+self.Serialize(root.right)
    #递归反序列化
    def Deserialize(self, s):
        self.flag += 1
        l = s.split(',')
        if self.flag >= len(s):
            return None
        if l[self.flag] != '#':#非空
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        else:
            root = None #这是空节点
        return root
```
#### 题目62：二叉搜索树的第K个结点
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
```python
"""
中根序遍历 到第k个即可返回
"""
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        if not pRoot or k==0: return None
        stack = []
        index = 0
        while pRoot or stack:
            while pRoot:
                stack.append(pRoot)
                pRoot = pRoot.left
            if stack:
                pRoot = stack.pop()
                index += 1
                if (index==k):return pRoot
                pRoot = pRoot.right
        return None
```
#### 题目63：数据流中的中位数
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
```python
"""
用一个列表存数据流，写时往里加，读时返回即可（ps,方法不够好，有待改进）
"""
class Solution:
    def __init__(self):
        self.nums = []
        self.length = 0
    def Insert(self, num):
        self.nums.append(num)
        self.nums.sort()#O(nlogn),时间复杂度有点高
        self.length += 1
    def GetMedian(self,emmm):
        if self.length%2 ==1:
            return self.nums[self.length//2]/1.0
        else:
            return (self.nums[self.length//2] + self.nums[self.length//2-1])/2.0
```
#### 题目64：滑动窗口的最大值
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
```python
"""
方法1：
用一个滑动窗口去记录，res记录最大值,每次窗口右移一位。
求最大值不需要每次都遍历窗口，可以把之前的最大值和最右边的一位比较，这样时间O（1）
当然，如果窗口移动到了前一个最大值的后面，那么应该重新计算最大值。
所以整体的时间是O(n)，当然最差有可能达到O（SIZE*n）
方法2：
利用队列，比前一种滑动窗口快一些，O(n)，并且最差也是O(n)
"""
class Solution:
    def maxInWindows(self, num, size):
        #方法1
        """
        # 首先是鲁棒性检测
        if not num or size ==0 or size>len(num): return []
        low = 0
        high = size-1
        res = []
        length = len(num)
        maxnum = max(num[low:high+1])
        flag = 0
        while high<length:
            if num[high]>maxnum:
                maxnum = num[high]
                flag = high #flag记录当前窗口最大值的位置
            if low>flag:
                maxnum = max(num[low:high+1])#当窗口不包括前一个最大值时，需重新计算最大值
            res.append(maxnum)
            low += 1
            high += 1
        return res
        """
        
        #方法2：利用队列，比前一种滑动窗口快一些，O(n)，并且最差也是O(n)
        queue,res,i = [],[],0
        while size>0 and i<len(num):
            if len(queue)>0 and i-size+1 > queue[0]: #若最大值queue[0]位置过期 则弹出 
                queue.pop(0)
            while len(queue)>0 and num[queue[-1]]<num[i]: #弹出所有比num[i]小的数字
                queue.pop()
            queue.append(i)
            if i>=size-1:
                res.append(num[queue[0]])
            i += 1
        return res
```
#### 题目65：矩阵中的路径
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
```python
"""
思路：
1.首先是在矩阵（实际是一维数组）matrix中找到第一个与path同字符的位置(i,j)
2.然后对这个位置进行判断：
    如果这个位置到了矩阵外，或者当前字符不等于path里的字符，或者之前已经经过了这个位置，返回False
    如果此时记录长度的K已经等于path的长度，则说明路径匹配成功，返回True
    当前位置成功了后，递归判断其上下左右四个位置是否满足路径的条件(成功返回True,都不成功，换一个起点重新开始)
"""
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        
        #help函数，递归判断路径是否满足条件
        def judge(mat,i,j,rows,cols,flag,strs,k):
            index = i*cols+j
            if(i<0 or i>=rows or j<0 or j>=cols or mat[index]!=strs[k] or flag[index] == True):
                return False
            if k == len(strs)-1:#若k已经到达strs末尾，说明之前的都已经匹配成功了，直接返回true即可
                return True
            flag[index] = True #要走的第一个位置置为true，表示已经走过了
            if (judge(mat,i-1,j,rows,cols,flag,strs,k+1)
                or judge(mat,i+1,j,rows,cols,flag,strs,k+1)
                or judge(mat,i,j-1,rows,cols,flag,strs,k+1)
                or judge(mat,i,j+1,rows,cols,flag,strs,k+1)):
                return True
            flag[index] = False #走到这，说明这一条路不通，还原，再试其他的路径
            return False
        
        #遍历矩阵中的每个字符，找到一个字符开头，然后递归
        flag = [False]*len(matrix)
        for i in range(rows):
            for j in range(cols):
                if judge(matrix,i,j,rows,cols,flag,path,0):
                    return True
        return False
```
#### 题目66：机器人的运动范围
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
```python
"""
此题与上一个类似，均可以通过递归的方式求解。
需要用flag记录一下能达到的格子，Count函数统计能达到的格子的数目，sumBybit函数计算位数字和。
"""
class Solution:
    #help函数，计算位数字和
    def sumByBit(self,i):
        bitsum = 0
        while i !=0:
            bitsum += i%10
            i = i/10
        return bitsum
    
    #这个是递归的函数，把能达到且位数字和不超过限制的格子标记出来，且格子数加1
    def Count(self,threshold,i,j,rows,cols,flag):
        index = i*cols+j
        if (i<0 or i>=rows or j<0 or j>=cols 
            or flag[index] == True 
            or (self.sumByBit(i)+self.sumByBit(j)>threshold)):
            return 0
        flag[index] = True
        return (self.Count(threshold,i-1,j,rows,cols,flag)
                +self.Count(threshold,i+1,j,rows,cols,flag)
                +self.Count(threshold,i,j-1,rows,cols,flag)
                +self.Count(threshold,i,j+1,rows,cols,flag)
                +1)
    
    #主函数调用递归函数
    def movingCount(self, threshold, rows, cols):
        flag = [False]*(rows*cols) #标记某个位置是否到达过
        return self.Count(threshold,0,0,rows,cols,flag)
```

### 《剑指Offer》第六期：就到这里啦，感谢您的收藏与转发。
《剑指Offer》专题终于完成，但不意味着结束，以后可能对某些题的答案进行更新，感谢各位的支持~ 
### 完结撒花



