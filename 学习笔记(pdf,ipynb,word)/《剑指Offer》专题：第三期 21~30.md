### 《剑指Offer》专题: 第三期 21-30
这里是《剑指Offer》专题系列第三期，一共十题，内容大约需要60分钟阅读，对应牛客网序号。如有任何疑问或者你有更好的方法或改进，欢迎#投稿#或者到github中提交PR.
#### 投稿方式在公众号右下角，联系我们处。

题目21：栈的压入、弹出序列
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
```python
# -*- coding:utf-8 -*-
"""
方法1 和方法2都能通过，但方法1时间复杂度是O（n^2），虽然其主体部分是O（n）的
方法2 时间复杂度为O（n）,因此更快。
"""
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        
        """
        
        #方法1： O（n^2）
        stack = [] #模拟进栈和出栈的过程
        
        #这一段是判断栈里是否有不同元素，有不同元素，返回False  O（n^2）
        for c in popV:  
            if c not in pushV:
                return False
            
        #模拟第一个元素出栈前的压栈动作#O(n)
        x = pushV.index(popV[0])
        for i in range(x+1):
            stack.append(pushV[i])
        
        #模拟所有元素的出栈动作（还有进栈）#O(n)
        for ch in popV:
            if ch not in stack:
                stack.append(ch)
                stack.pop()
            if ch == stack[-1]:
                stack.pop()
            if ch in stack and ch != stack[-1]:
                return False
        
        return True #如果中间没有False，则返回True
        """
        
        #方法2：时间O(n)
        #不需要一开始判断是否两者有不同的元素，如有不同元素，会不满足while， stack就非空
        if not pushV:
            return False
        stack = []
        j = 0
        for i in range(len(pushV)): #这个方法是O(n)时间的
            stack.append(pushV[i]) #全部入栈
            while j < len(popV) and stack[-1] == popV[j]:#遇到栈顶和popV[j]相等才出栈
                stack.pop()
                j += 1
        return  not stack #如果栈为空，说明可以全部pop()掉，因此返回True
```
题目22：从上往下打印二叉树
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
```python
# -*- coding:utf-8 -*-
"""
此题：此题就是二叉树的广度优先遍历，同时需要保存一个值的列表。
"""
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        
        # BFS，广度优先
        if not root:return []
        queue = [root]
        res = [] 
        while queue:
            curr = queue.pop(0)#深度优先stack.pop()，广度优先queue.pop(0)
            res.append(curr.val)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return res
```
题目23：二叉搜索树的后续遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
```python
# -*- coding:utf-8 -*-
"""
二叉搜索树的特征：每个结点左子树的所有值都小于当前结点值；右子树都大于当前结点值
对应后序序列特征：
    1.根结点位一个序列的最后一个元素；
    2.然后序列前部分所有小于根结点值的属于左子树，并且最后一个小于根结点的是左子树的根；
    3.后半部分属于右子树，并且最后一个大于根结点的是右子树的根；
    
所以思路可以为：
    1.先找到整棵树的根结点（sequence[-1]）
    2.通过遍历sequence[i],与sequence[-1]比较大小，找到左子树的根结点left_s
    3.判断sequence后面的每一个元素是否都大于根结点（不大于就返回False）
    
    4.递归：
    然后对左子树、右子树 递归该过程，判断是否满足二叉搜索树特征，只有两个都为True，才返回True

参考链接：https://cuijiahua.com/blog/2017/12/basis_23.html
"""
class Solution:
    def VerifySquenceOfBST(self, sequence):
        
        #鲁棒性代码段，也是递归退出的代码段
        if not len(sequence):
            return False
        if len(sequence) == 1:
            return True
        
        #迭代的起点
        length = len(sequence)
        root = sequence[-1] #根一定为后序遍历最后一个
        i = 0
        
        #找到分界点（即当前根结点的左右子树的分界点），并判断是否满足二叉搜书树特征
        #特征：根结点在后序遍历列表末尾，树中根结点左边全小于它（在遍历序列的前部分），反之在后部分
        while sequence[i] < root:
            i += 1
        k = i #从K位置开始的都应大于当前根
        for j in range(i, length-1):
            if sequence[j] < root: #如果不满足从k开始的都大于当前根
                return False #不满足二叉搜索树的条件
        
        #需要递归继续判断是否是二叉搜索树的前后两个序列（以sequence[k]为分界（后序列的开始））
        left_s = sequence[:k]
        right_s = sequence[k:length-1]
        
        #递归过程 当左右子树都是二叉搜索树，整棵树才是二叉搜索树
        left, right = True, True
        if len(left_s) > 0:
            left = self.VerifySquenceOfBST(left_s)
        if len(right_s) > 0:
            right  = self.VerifySquenceOfBST(right_s)
        return left and right #全为True才为True
```
题目24：二叉树中和我某一值的路径
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
```python
"""
python版
方法1：深度优先  非递归版
方法2：广度优先
方法3：深度优先  递归版

注：此题还要求返回list按数组长度大小排序，因此要对res排序

注：此题与LeetCode 122题类似，122题只要求判断是否存在给定值的路径和，而不需要返回路径。
那么，我们可以使用122题类似的做法，并新增一项记录路径，以及路径的列表。
LeetCode122题答案可以见我之前的文章《数据结构与算法 学习笔记(7):二叉树和树》：题目8
https://blog.csdn.net/LSayhi/article/details/88783857
"""
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        
        """方法1：深度优先 非递归版""" 
        
        
        #鲁棒性代码段
        if not root:
            return []
        
        #三元组记录：(根结点，当前路径和，当前路径)；LeetCode 122题只需二元组
        stack = [(root,root.val,[root.val])]
        res =[]
        
        #深度优先遍历的基本操作
        while stack:
            cur, sums, route = stack.pop() #当前结点，当前路径和，当前路径
            if not cur.left and not cur.right:#判断是否是叶结点
                if sums == expectNumber:#判断路径和是否等于给定值
                    res = res + [route]#将等于给定值的路径加入到res中
            if cur.right:#
                x = cur.right.val
                route_right=route[:]#深拷贝，否则左分支使用route时，父节点route已加上右分支的结点
                route_right.append(x)#在深拷贝上append，不改变原route的值，左分支使用route不受影响
                stack.append((cur.right,sums + x,route_right))#stack继续压入三元组
            if cur.left:#左分支操作同右分支
                y = cur.left.val
                route_left=route[:]
                route_left.append(y)
                stack.append((cur.left,sums + y,route_left)) 
        return sorted(res,key = lambda x:-len(x)) #二维数组按长度从大到小排列
        
        
        """方法2：广度优先"""
        
        """
        #鲁棒性代码段
        if not root:
            return []
        
        #三元组记录：(根结点，当前路径和，当前路径)；LeetCode 122题只需二元组
        queue = [(root,root.val,[root.val])]
        res =[]
        
        #广度优先遍历的基本操作
        while queue:
            cur, sums, route = queue.pop(0) #当前结点，当前路径和，当前路径
            if not cur.left and not cur.right:#判断是否是叶结点
                if sums == expectNumber:#判断路径和是否等于给定值
                    res = res + [route]#将等于给定值的路径加入到res中
            if cur.right:#
                x = cur.right.val
                route_right=route[:]#深拷贝，否则左分支使用route时，父节点route已加上右分支的结点
                route_right.append(x)#在深拷贝上append，不改变原route的值，左分支使用route不受影响
                queue.append((cur.right,sums + x,route_right))#stack继续压入三元组
            if cur.left:#左分支操作同右分支
                y = cur.left.val
                route_left=route[:]
                route_left.append(y)
                queue.append((cur.left,sums + y,route_left)) 
        return sorted(res,key = lambda x:-len(x)) #二维数组按长度从大到小排列
        """
        
        
        """方法3：深度优先 递归版""" 
        """
        #鲁棒性代码段
        if not root:
            return []
        
        if not root.left and not root.right:
            if expectNumber == root.val:
                return [[root.val]]
        
        #递归段
        tmp = self.FindPath(root.left, expectNumber-root.val) + self.FindPath(root.right, expectNumber-root.val)
        return sorted([[root.val]+i for i in tmp],key = lambda x: -len(x))
        """
```
题目25：复杂链表的复制
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
"""
时间复杂度O(n):遍历n个结点
空间复杂度O(n)：每遍历一个结点，都需要新建一个temp结点

题目要求：要求复制链表，并把新链表的首结点返回（注意不是引用）。
实现效果：深拷贝复制得到新链表，并且未改变原表。

思路：
    首先是建一个指针q来遍历原链表
    建一个结点last指向新表表末，再用一个结点指向新表last前一点
    遍历原链表，复制每个结点：每次新建结点，并把原节点的值和指针赋值给新结点
    复制一个结点后，把原链表遍历指针后移
    把新结点的末尾用指针标记，然后继续遍历下一个
"""
import copy
class Solution:
    def Clone(self, pHead):
        # 鲁棒性代码
        if not pHead:return None
        
        #这里新建两个结点，res指pHead前一个结点，dummy为last前一结点
        q = pHead
        last = RandomListNode(0)
        dummy = RandomListNode(0)
        dummy.next = last
        
        #遍历结点，每次新建temp结点保存q结点的值和指针域，添加到last后（不能引用）
        while q:
            temp = RandomListNode(q.label) #这里的四句是为了深拷贝
            #temp.next = copy.deepcopy(q.next)
            #temp.random = copy.deepcopy(q.random)
            next_ = RandomListNode(0)
            random_ = RandomListNode(0)
            next_ = q.next
            random_ = q.random
            temp.next = next_
            temp.random = random_
            last.next = temp
            
            q = q.next #向后移一位
            last = last.next #last总指向复制链表最后一个结点
        return dummy.next.next #返回复制链表的首结点
```
题目26：二叉搜索树与双向链表
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
```python
"""
此题可采用递归思想：
递归中序遍历，右->中->左
首先可以定义一个引用self.res，记录的是双向链表的头结点，初始化为空；
0.日常鲁棒性检测
1.递归右子树
2.把根结点加入到递归得到的res之前，更新res
3.递归左子树
这个过程相当于一直按从大到小的顺序把结点链接起来(小的在前，大的在后).
最后最前面的就是要返回的双向链表首结点。
"""
class Solution:
    def __init__(self):
        self.res = None
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:return pRootOfTree
        
        self.Convert(pRootOfTree.right)
        if self.res == None:
            self.res = pRootOfTree
        else: #双向链接
            self.res.left = pRootOfTree
            pRootOfTree.right = self.res
            self.res = pRootOfTree
        self.Convert(pRootOfTree.left)
        
        return self.res
```
题目27：字符串的排列
输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
```python
# -*- coding:utf-8 -*-
"""
此题为全排列问题：
    指向第0位，把它和位置在它后面（包括自己）所有的位置交换，得到一波排列；
    指向第1位，把它和位置在它后面（包括自己）所有的位置交换，得到一波排列；
    指向第2位....
    如果指向了最后一位，则返回得到的排列
    是一个在循环中递归的问题
    注意：此题输入是字符串，输入要转为数组后才能调用全排列函数，全排列的输出要转回字符串
    注意：要避免 重复项的产生 如 输入"aa" 应输出["aa"],而不是["aa","aa"]
参考链接:https://blog.csdn.net/qq_31601743/article/details/82053201
"""
class Solution:
    def __init__(self):
        self.res =[] #返回的结果
    def Permutation(self,ss):
        #全排列函数
        def permutations(arr,position,end):
            if position == end:
                temp = ''.join(arr) #这句是为了把数组转成字符串存在列表res中
                self.res.append(temp)
            for index in range(position, end):
                #if是去掉全排列中的重复项，比如"aa",会输出["aa","aa"],本应输出["aa"]
                if index == position or arr[index]!=arr[position]:
                    arr[index], arr[position] = arr[position], arr[index]#交换
                    permutations(arr, position+1, end)#递归
                    arr[index], arr[position] = arr[position], arr[index]#恢复
        #主体部分
        if not ss:return []
        s = [i for i in ss] #先把字符串转为数组操作，否则不能交换字符串中的值
        permutations(s,0,len(s)) #调用自定义全排列函数
        return sorted(self.res)#其实返回res就行，但牛客网测试结果是按字典序判断的
```
题目28：数组中出现次数超过一半的数字
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
```python
# -*- coding:utf-8 -*-
"""
方法1：时间O（n）,空间O(n)
    建立字典统计数字出现的次数，如果大于一半，则返回该数字
方法2：时间O（n）,空间O（1）
    用两个变量存数字和次数；遍历数组，如果下一位数字和变量一样，则次数+1，否则-1.
    如果次数为0，则把变量赋值为当前遍历的数字，且次数赋值为1，然后继续遍历。
    如果最后次数>1，说明有一个数字次数大于一半（且就是变量保存的数字），返回该数字
"""
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        
        # 方法1： 
        """
        if not numbers:
            return 0
        dic = {}
        length = len(numbers)
        res = 0
        for ch in numbers:
            if ch not in dic:
                dic[ch] = 1
            else:
                dic[ch] += 1
            if dic[ch] > length//2:
                return ch
        return res
        """
        
        #方法2：时间O（n）,空间O(1)
        res = numbers[0]
        freq = 1
        if not numbers:return 0
        for num in numbers:
            if num == res:
                freq += 1
            if num != res:
                freq -= 1
            if freq == 0:
                res = num
                freq = 1
        if freq > 1:
            return res
        else:
            return 0
```
题目29：最小的K个数
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
```python
# -*- coding:utf-8 -*-
"""
方法1：快速排序or蒂姆排序， O(nlogn),O(1)，适中
方法2：建立大顶堆 O（nlogk）,O(k)，适合海量数据！不改变原数组，用O(k)小空间换取时间的大大降低
方法3：利用parition思想： O（n）,O(1) ，时间复杂度最低，但会改变原有的数组，且前K个数字未排序
方法4：最直接的两层循环 O（kn）最坏O（n^2），不合适，不去实现

"""
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        
        #方法1：直接排序 时间 nlogn，空间O（1）
        """
        if not tinput or k<=0 or k > len(tinput)+1:
            return []
        return sorted(tinput)[:k]"""
        
        #方法2：基于大顶堆或红黑树 时间 O（nlogk）,空间O（k）
        n = len(tinput)
        if k <= 0 or k > n or n == 0:
            return []
        # 建立大顶堆
        for i in range(k //2 - 1, -1, -1):
            self.heapAjust(tinput, i, k - 1)
        for i in range(k, n):
            if tinput[i] < tinput[0]:
                tinput[0], tinput[i] = tinput[i], tinput[0]
                # 调整前k个数
                self.heapAjust(tinput, 0, k - 1)
        return sorted(tinput[:k])#这里本来是不用sorted的，但牛客网竟然是要排序的结果
    def heapAjust(self, nums, start, end):
        temp = nums[start]
        # 记录较大的那个孩子下标
        child = 2 * start + 1
        while child <= end:
            # 比较左右孩子，记录较大的那个
            if child + 1 <= end and nums[child] < nums[child + 1]:
                # 如果右孩子比较大，下标往右移
                child += 1
            # 如果根已经比左右孩子都大了，直接退出
            if temp >= nums[child]:
                break
            # 如果根小于某个孩子,交换两者，将较大值提到根位置
            nums[start], nums[child] = nums[child], nums[start]
            # 接着比较被降下去是否符合要求，此时的根下标为原来被换上去的那个孩子下标
            start = child
            # 孩子下标也要下降一层
            child = child * 2 + 1
        
        # 方法3：时间 O（n），空间O（1），但返回K个数字不一定按顺序，且会修改原数组
        """
        def parition_TwoPoint(array,begin,end):
            pivot = array[begin]#枢轴
            while begin < end:#两个指针begin、end
                while(begin < end and array[end] >= pivot):#end从右向左
                    end -= 1 # 一直向左移，其它不动
                array[begin] = array[end] #当array[end]<= pivot,将其赋值给begin位置（即小的放左边去）
                while(begin < end and array[begin]<= pivot):
                    begin += 1 #一直向右移
                array[end] = array[begin] #当array[begin]> pivot,将其赋值给end位置（即大的放右边去）
            pos = begin # 当begin == end，此时的位置即是pivot应放入的位置
            array[pos] = pivot
            return pos
        
        #主体
        begin, end = 0, len(tinput)-1
        if not tinput or k<=0 or k > end+1:
            return []
        k_numlst = []
        while begin< end:
            pos = parition_TwoPoint(tinput,begin,end)
            if pos == k-1:#如果pos==k-1，刚好
                k_numlst = tinput[0:pos+1]
                break
            elif pos > k -1:#如果pos大于k-1，说明真正的pos在左边
                end = pos
            else:#如果pos小于k-1，说明真正的pos在右边
                begin = pos + 1
        return k_numlst
        """
```
题目30：连续子数组的最大和
HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
```python
# -*- coding:utf-8 -*-
"""
典型的动态规划问题：时间O(n),空间O（n）
此题动态规划的递推公式为： dp[i] = max(dp[i-1]+array[i],array[i])

方法2是对方法1的改进，时间复杂度降低约一半，但还是O（n）,空间不变
"""
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # 动态规划
        
        #方法1：
        """
        n = len(array)
        dp = [0]*n #dp[i]中存的是以i位置结尾的连续序列的和的最大值，初始化为0
        dp[0] = array[0] #dp[0]= array[0],是动态规划的起始条件
        for i in range(1,n):#依次计算dp[i]
            dp[i] = max(dp[i-1]+array[i],array[i])
            #此题中动态规划的公式
        return max(dp) #从所有的最大子序列和dp[i]中再找最大的
        """
        #方法2：
        n = len(array)
        dp = [0]*n #dp[i]中存的是以i位置结尾的连续序列的和的最大值，初始化为0
        dp[0] = array[0] #dp[0]= array[0],是动态规划的起始条件
        maxdp = dp[0]
        for i in range(1,n):#依次计算dp[i]
            dp[i] = max(dp[i-1]+array[i],array[i])
            if dp[i]> maxdp:
                maxdp=dp[i]
            #此题中动态规划的公式
        return maxdp #从所有的最大子序列和dp[i]中再找最大的
```
### 《剑指Offer》第三期：就到这里啦，感谢您的收藏与转发。
####<center>更多精彩，可访问往期文章<center>
有任何疑问，欢迎后台留言，如果对文章内容有更深或独特的见解，欢迎#投稿#或者到github中PR。本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork.

github链接：https://github.com/LSayhi
代码仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai

![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-885f1ec27454b67a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)