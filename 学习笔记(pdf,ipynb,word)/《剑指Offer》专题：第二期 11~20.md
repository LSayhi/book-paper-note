### 《剑指Offer》专题: 第二期11~20
这里是《剑指Offer》系列文章第二篇，一共十题，内容大约需要90分钟阅读理解。记录书中所有题目的求解方法以及考察点，如果大家有任何疑问，欢迎后台留言，如果有更好的方法或者改进，欢迎投稿或者到github中PR。
注：这里的题目顺序是牛客网中《剑指Offer》默认序，和书中的略有区别。
##### 题目11：二进制中1的个数
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
```python
# -*- coding:utf-8 -*-
"""
此题考查的是位运算，见方法1和方法2的注释。
"""
class Solution:
    def NumberOf1(self, n):
        # write code here
        """
        #方法1：
        count, flag = 0, 0
        if n < 0: #对负数取32位，并把最高位变成0，避免 while n 出现死循环
            n = n & 0x7fffffff
            flag = 1
        while n:
            n = (n - 1) & n#与的结果是把n的二进制右起第一个1变成0
            count += 1 #有多少1就统计多少次
        if flag: #如果n是负数 需要增加补码开头的那个1
            return count+1
        else: #如果n是正数，则正常统计
            return count
        """
        #方法2：
        count = 0
        if n < 0: #对负数取32位，就变成是正数了，避免了 while n 出现死循环
            n = n & 0xffffffff
        while n:
            n = (n - 1) & n#与的结果是把n的二进制右起第一个1变成0
            count += 1 #有多少1就统计多少次
        return count
```
##### 题目12：数值的整数次方
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
```python
# -*- coding:utf-8 -*-
"""
思路：快速降幂
if n是奇数： # if n & 1 == 1,这样比 if n % 2 ==1要快
    a^n = a(n/2)*a(n/2)*a 
if n是偶数：
    a^n = a(n/2)*a(n/2) 
注意特殊情况，base == 0；exponent ==0;exponent == -1;后两个递归退出条件
"""
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0: #处理底数为0的情况
            return 0
        if exponent == 0: #递归终止条件
            return 1
        if exponent == -1: #当exponent<0，exponent//2最终会是-1，而不是0（因为-1//2 == -1）
            return 1/base
        temp = self.Power(base,exponent//2)
        
        #快速降幂
        if exponent & 1:#n为奇数
            return temp*temp*base
        else:
            return temp*temp
```
##### 题目13：调整数组顺序使奇数位于偶数前面
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```python
# -*- coding:utf-8 -*-
"""
方法1：时间复杂度O（n）,空间复杂度O（n）
    利用一个新数组copy保存array的值，统计copy中奇数的个数
    然后在array前count_odd项保存奇数，后面的保存偶数
方法2：平均/最坏时间复杂度O（nlogn）,最好时间复杂度O（n）,最大空间复杂度O（n/2）
    利用python内置函数sort，其内部实现是Timsort
    Tim排序是稳定的，能保持相对位置不变，且具有适应性，最佳时间复杂度O(n)
    也就是在最好的情况下，时间复杂度和方法1一样，空间复杂度比方法1小一半（虽然同O（n））
其它方法：
    例如：冒泡、插入具有稳定性，空间复杂度O（1），但时间复杂度O（n^2）
    堆排序具有稳定性，但是时间复杂度O（nlogn）,空间复杂度O（n）,不如方法1和方法2
"""
class Solution:
    def reOrderArray(self, array):
        # write code here
        
        #方法1
        """
        count_odd = 0
        copy=[0]*len(array)
        for i in range(len(array)):
            copy[i] = array[i]
            if copy[i] & 1:
                count_odd += 1
        i, j = 0,count_odd
        for ch in copy:
            if ch & 1:
                array[i] = ch
                i += 1
            else:
                array[j] = ch
                j += 1
        return array 
    """
        #方法2：python内置sorted函数（Timsort）
        array.sort(key =lambda x: x%2==0)
        return array
```
##### 题目14：链表中倒数第k个结点
输入一个链表，输出该链表中倒数第k个结点。
```python
"""
方法1： Time = O(n), Space = O(1)
先遍历统计表长，记为count；
然后从表头走count - k 步，到达倒数第K个，返回之，当然要注意边界条件

方法2：
    两个指针，p2先走K-1步，然后p1和p2一起走，P2到表末，P1到达倒数第K位，返回之
    其实和方法1是类似的，时间复杂度相同，空间复杂度也相同。
"""
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        
        #方法1：
        """
        if not head or k <= 0:#特殊情况
            return None
        p, q = head, head.next
        count = 1
        while q: #计算整个表长度
            count += 1 
            q = q.next
        if count < k: #如果k大于表长，返回None
            return None
        else: # 正常情况，则从表头走 count - k 步
            i = 1
            while i < count-k+1:
                p = p.next
                i +=1
        return p
        """
        
        #方法2：
        if head==None or k<=0:
            return None
        p2=head
        p1=head
        #p2先走，走k-1步，如果k大于链表长度则返回 空，否则的话继续走
        while k>1:
            if p2.next!=None:
                p2=p2.next
                k-=1
            else:
                return None
        #两个指针一起 走，一直到p2为最后一个,p1即为所求
        while p2.next!=None:
            p1=p1.next
            p2=p2.next
        return p1
```
##### 题目15：反转链表
输入一个链表，反转链表后，输出新链表的表头。
```python
"""
反转链表，老生常谈了。
依次摘下原表头，放在新表头p的前面，更新的新表头p
"""
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        p = pHead
        q = pHead.next
        p.next = None
        while q:
            temp = q.next
            q.next = p
            p = q
            q = temp
        return p
```
##### 题目16：合并两个排序的链表
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
```python
"""
思路：时间复杂度O（n + m）,空间复杂度O（1），m和n为两表表长
0.鲁棒性代码，判断是否有空链表。
1.创建虚拟表头，dummy，并用一个p结点记录dummy的位置（p.next = dummy）
2.用两个指针分别指向phead1和pHead2的表头，
  当两个表都没到表末时，比较两个位置值的大小，小的就放到合成链表后，值小的指针向后移
3.如果有一个指针已经到表末了，那么就把另一个表剩余的结点依次加到合成表后。
4.返回真正的链表头，p.next.next
"""
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        
        # pHead1 或 pHead2为空，鲁棒性代码
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        
        #dummy是新合成链表的假表头，p是dummy的前一结点，是为了记录表头而创建的
        #所以p.next.next即为合成的链表的表头结点
        dummy = ListNode(0)
        p = ListNode(0)
        p.next = dummy
        
        #判断两个表中的元素大小，哪个表的值小就把哪个放到合成表后，并指向下一位继续判断
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                dummy.next = pHead1
                dummy = dummy.next
                pHead1 = pHead1.next
            else:
                dummy.next = pHead2
                dummy = dummy.next
                pHead2 = pHead2.next
        
        #两个链表长度不一致，把其中一个表剩余的部分添加到合成表后
        while pHead1:
            dummy.next = pHead1
            dummy = dummy.next
            pHead1 = pHead1.next
        while pHead2:
            dummy.next = pHead2
            dummy = dummy.next
            pHead2 = pHead2.next
        return p.next.next #返回真正的合成表的表头
```
##### 题目17：树的子结构
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
```python
"""
代码主体部分：

    首先是判断树A和树B是否是空树，若其中至少一个为空，返回False
    其次，判断A和B的根结点是否相同
    如果根节点相同，则判断树A、树B 是否相同（这里的判断需要自定义一个函数issame,返回值是True和False）
    如果根节点不同，则判断树A的左子树是否包含树B(递归实现)，若是，result == True
    若不是，则判断A的右子树是否包含树B（递归实现），若是,result == True
    
    以上过程，若都没有成功判定B是A的子结构，则返回result == 默认值False，否则，result== True

注意：一种先把两棵树遍历，转成字符串，然后判断B树字符串是否在A中，这种做法是错误的（可以举反例）
如：Node(1,Node(2,Null,3),Null) 和 Node(1,2,3)产生的字符串是一样的，但是他们不是一种结构
"""

class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        
        # 判断两棵树是否相同的函数
        def issame(tree1,tree2):
            if tree2 == None:
                return True
            if tree1 == None:
                return False 
            if tree1.val != tree2.val:
                return False
            return issame(tree1.left,tree2.left) and issame(tree1.right,tree2.right)
        
        #主体部分
        
        #鲁棒性代码
        if not pRoot1 or not pRoot2:
            return False
        
        #返回值，初始化
        result = False
        
        if pRoot1.val == pRoot2.val:#若结点值相等，判断两者为根的树是否是相同
            result = issame(pRoot1,pRoot2) #result可能为True 或False
        if result == False: #如果上面result为False,则递归判断左子树是否包含pRoot2
            result = self.HasSubtree(pRoot1.left,pRoot2)
        if result == False: #如果上面result为False,则递归判断右子树是否包含pRoot2
            result = self.HasSubtree(pRoot1.right,pRoot2)
        return result #返回最终的result
```
##### 题目18：二叉树的镜像
操作给定的二叉树，将其变换为源二叉树的镜像。
```python
"""
镜像就是把每个结点的左右子树交换，因此递归就可以。
"""
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return None
        root.left,root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root
```
##### 题目19：顺时针打印矩阵
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
```python
# -*- coding:utf-8 -*-
"""
分为两个步骤：
1. 打印一圈
    从左到右打印一行（这是一定有的）
    当有两行及以上时，存在从上到下打印
    当有两行及以上并有两列及以上时，存在从右到左
    当有两列并有三行及以上时，存在从下到上打打印
2. 起点进入下一圈，即进入下一个循环
    初始几点为(0, 0), 打印一圈后有(1, 1), 再打印一圈为(2, 2)
    都存在行数 > 起点 * 2 并且列数 > 起点 * 2
    简单来说就是,如果把二维数组等分为四个部分,中点都为中心，那么起点肯定都位于左上部分
    根据这个条件可以判断是否还有圈数。
"""
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        def printMatrixCircle(start):
            endRow = rows - 1 - start
            endColumn = columns - 1 - start
            # 从左到右打印一行
            for y in range(start, endColumn + 1):
                result.append(matrix[start][y])
            # 从上到下打印一列
            if endRow > start:
                for x in range(start + 1, endRow + 1):
                    result.append(matrix[x][endColumn])
            # 从右到左打印一行
            if endColumn > start and endRow > start:
                for y in range(endColumn - 1, start - 1, -1):
                    result.append(matrix[endRow][y])
            # 从下到上打印一行
            if endRow > start + 1 and endColumn > start:
                for x in range(endRow - 1, start, -1):
                    result.append(matrix[x][start])
        # write code here
        if not matrix:
            return []
        columns = len(matrix[0])
        rows = len(matrix)

        start = 0
        result = []
        while columns > start * 2 and rows > start * 2:
            printMatrixCircle(start)
            start += 1
        return result
```
##### 题目20：包含min函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
```python
# -*- coding:utf-8 -*-
"""
用两个栈，一个栈维护 标准栈
另一个辅助栈用来维护min列表，以保证返回min操作时间是O（1）的

注：push和pop操作需要同时对标准栈和辅助栈操作
"""
class Solution:
    def __init__(self):
        self.stack = [] # 栈保存值
        self.assist = [] #辅助栈保存当前最小值的列表
         
    def push(self, node): #push是O（n）操作，因为有min()函数
        min = self.min()
        #把当前最小值加入辅助栈
        if not min or node < min:
            self.assist.append(node) 
        else:
            self.assist.append(min)
        self.stack.append(node) #把当前值加入栈 
         
    def pop(self): #pop()操作要pop()数据栈，也要pop()辅助栈
        if self.stack:
            self.assist.pop()
            return self.stack.pop()
    def top(self):
        # write code here
        if self.stack:
            return self.stack[-1]
         
    def min(self): # 返回辅助栈栈顶元素 O（1）
        # write code here
        if self.assist:
            return self.assist[-1]
```
### 《剑指Offer》专题：第二期就到这里啦，感谢您的收藏与转发。
####<center>更多精彩，可访问以下链接<center>

本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork.

github链接：https://github.com/LSayhi
代码仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai

![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-885f1ec27454b67a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)