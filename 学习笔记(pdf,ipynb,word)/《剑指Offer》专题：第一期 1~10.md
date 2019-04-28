### 《剑指Offer》专题: 第一期1~10
话说东汉末年,群雄割据,盗贼蜂起。对不起，走远了，走远了。继上个系列（数据结构与算法）基础连载之后，这次我将推出《剑指Offer》系列文章，记录书中所有题目的求解方法以及考察点，如果大家有任何疑问，欢迎后台留言，如果有更好的方法或者改进，欢迎投稿或者到github中PR。
注：这里的题目顺序是牛客网中《剑指Offer》默认序，和书中的略有区别。
##### 题目1：二维数组中的查找
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```python
# -*- coding:utf-8 -*-
"""
假设二维数组有m个长度为n的一维数组，
思路1：
    二维数组，即列表的列表，由于每一行都序，我们可以按行读取，然后二分查找
    
    时间复杂度： 二分O(log n),遍历所有行，所以O(mlog n)
思路2：
    思路1 只利用了 行的单调性，此题 列也有单调性.
    把二维数组想象成矩阵，当taeget < array[i][j]时，target会小于array[i][j]右下方所有值
    因此，我们可以从二维数组的左下角出发遍历，比较其和target之间的大小
    若 target更大，遍历位置向右移一位（j++）,反之，向上移（i--）,若相等，返回True
    
    时间复杂度: <= O(m+n),每次移一位，最多只能移 m+n 次(到右上角),再多就移出了
"""
class Solution:
    # array 二维列表
    def Find(self, target, array):
        
        """
        思路2 ： 利用更多条件
        """
        if not array:
            return False
        n = len(array[0])-1
        m = len(array)-1
        i = m
        j = 0
        while i >= 0 and j <=n:
            if array[i][j] < target:
                j += 1
            elif array[i][j] > target:
                i -= 1
            else:
                return True
        return False
            
        #思路1 二分法
        """
        if not array:
            return False
        for arr in array:
            low, high = 0, len(arr)-1
            while low <= high: #二分查找
                mid = (low + high)//2
                if arr[mid] > target:
                    high = mid - 1 
                elif arr[mid] < target:
                    low = mid + 1
                else:
                    return True
        return False
        """
```
##### 题目2：替换空格
请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
```python
# -*- coding:utf-8 -*-

"""
方法1： 新建一个字符串，遍历原串，不断添加。
       时间 O(n),空间O(n)
方法2： 原地改，空间复杂度O（1）
       
"""
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        
        #方法1 ：牺牲空间换时间
        res = ""
        for char in s:
            if char != ' ':
                res += char
            else:
                res += "%20"
        return res
```
##### 题目3：从尾到头打印链表
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
"""
方法1： 反转链表，然后顺序访问存储在res中 
        注：此方法 改变了原链表，时间复杂度O（n）
方法2： 顺序访问链表，利用栈 先入后出的特点。
        注：无需改动原链表，时间复杂度O（n）,优于方法1
方法3： 遍历链表，存储在列表中，最后反转列表
        注：不改变原链表，时间复杂度O（n），较简单，就不给代码啦

"""
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        
        #方法2
        if not listNode:
            return []
        stack =[]
        res = []
        while listNode:
            stack.append(listNode.val)
            listNode = listNode.next
        while stack:
            res.append(stack.pop())
        return res
            
        # 方法1 先反转链表，再顺序存
        """
        if not listNode:
            return []
        p = listNode
        q = listNode.next
        p.next = None
        while q: #反转链表
            temp = q.next
            q.next = p
            p = q
            q = temp
        res = []
        while p:#依次读取反转后的链表
            res.append(p.val)
            p = p.next
        return res 
        """
```
##### 题目4：重建二叉树
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
"""
见注释
"""
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin: #鲁棒分析
            return None
        
        # prp.pop(0)一定是当前子树的根结点（一开始为整棵树的根节点）
        #pop(0)每次都弹出pre中的子树根结点，递归过程不会重复
        root = TreeNode(pre.pop(0))  
        index = tin.index(root.val) # 获得当前根结点的值在中序中的位置
        
        #子树递归
        #tin[:index]，因为tin中在index左边的都是当前根结点的左子树的结点
        #tin[index+1:]，因为tin中在index右边的都是当前根结点的右子树的结点
        root.left = self.reConstructBinaryTree(pre, tin[:index]) 
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root
```
##### 题目5：用两个栈来实现队列
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
```python
# -*- coding:utf-8 -*-
"""
定义栈Stack1和stack2，stack1看做是队列，stack2看做是辅助
1.push操作，只将Node放入stack1
2.pop操作，需要返回stack1的栈底（stack2的栈顶）,没有维持stack1的队列,只维持pop()返回值顺序
"""
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        self.stack1.append(node)
    def pop(self):
        if len(self.stack2) != 0:
            return self.stack2.pop()
        else:
            while len(self.stack1) !=0:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
```
##### 题目6：旋转数组最小的数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
```python
# -*- coding:utf-8 -*-
"""
方法1：顺序遍历，当找到第一个比前一位小的数时，返回之。时间复杂度O(n)
方法2：二分查找，时间复杂度O（logn）
    index1更新的目标是一步步指向未旋转时末尾的位置（例子中的5）；
    index2指向未旋转时的开头（例子中的1）；
    当index2 - index1 == 1时，index2的值就是要返回的值。
"""

class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        
        #方法1 ： 顺序查找，第一个小于前一位的就是最小数 时间复杂度O(n)
        """
        if not rotateArray:
            return 0
        cur = -1
        for ch in rotateArray:
            if ch >= cur:
                cur = ch
            else:
                return ch
        """
        #方法2 ： 二分查找 时间复杂度O(logn)
        if not rotateArray:#鲁棒性考虑
            return 0
        index1 = 0
        index2 = len(rotateArray)-1
        while rotateArray[index1] >=rotateArray[index2]: #二分
            if index2 - index1 == 1: #退出条件，index1指向旋转部分的末尾，index2指向未旋转的开头
                return rotateArray[index2]
            mid = (index1 + index2)//2
            if rotateArray[mid] >= rotateArray[index1]:
                index1 = mid
            else: #rotateArray[mid] < rotateArray[index1]:
                index2 = mid
```
##### 题目7：斐波那契数列
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0），n<=39
```python
# -*- coding:utf-8 -*-
"""
方法1：时间复杂度O（n）,空间复杂度O（n）
方法2：时间复杂度O（n）,空间复杂度O（1）
"""
class Solution:
    def Fibonacci(self, n):
        # write code here
        """ #时间复杂度O(N),空间复杂度O（N）
        if n == 0:
            return 0
        if n == 1:
            return 1
        res=[0,1] 
        while len(res)<=n:#时间
            res.append(res[-1]+res[-2]) #空间
        return res[-1]
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        res = [0,1] #空间复杂度O（1），时间复杂度O（n）
        for i in range(2,n+1): #时间复杂度O（N）
            temp = res[1]+res[0]
            res[0] = res[1]
            res[1] = temp
        return res[-1]
```
##### 题目8：跳台阶
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
```python
# -*- coding:utf-8 -*-
"""
此题思路：
第n阶台阶的到达方式有两种，
一种是从n-1阶跳一阶（剩下的n-1阶有f(n-1)种跳法）；
一种是从n-2阶跳俩阶（剩下的n-2阶有f(n-2)种跳法）；
所以，f(n) = f(n-1)+f(n-2),是个斐波那契问题，可以用斐波那契的解法。

这里，我们共有三种方式：
第一种是递归，由于时间复杂度太高，面试时我们不使用；
第二种是循环，这在斐波那契原始问题上已经使用，不作重复描述 T =O（n）,S = O（1）
第三种是使用公式，可以参考：https://blog.csdn.net/chichu261/article/details/83589767
用到了矩阵降幂，可以把时间复杂度将低到O（logn）,空间复杂度O（1）
"""
class Solution:
    def jumpFloor(self, number):
        # write code here
        def fib(n):
            if n < 1:
                return (1, 0)
            f_m_1, f_m = fib(n >> 1)
            if n & 1:
                return f_m_1 * (f_m_1 + 2 * f_m), f_m ** 2 + f_m_1 ** 2
            else:
                return f_m_1 ** 2 + f_m ** 2, f_m * (2 * f_m_1 - f_m)
        if number <=1:
            return number
        else:
            return fib(number)[0]
```
##### 题目9：变态跳台阶
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
```python
# -*- coding:utf-8 -*-
"""
跟青蛙跳台阶类似，第n阶台阶可以从n-1,n-2……1，0阶上起跳，
所以f(n)=f(n-1)+f(n-2)+……+f（n-(n-1)）+f(n-n)
写出前几项便可发现规律，n= 1,f(1)=1;n =2,f(2)=f(1)+f(1)=2;n>=2,f(n) = 2^(n-1)

如果计算2^(n-1)，我们可以用循环，每次*2，但是这样时间复杂度为O（n）；
如果用次方的快速降幂，则可以将时间降低到log(n),
因此可以自己写降幂或者直接利用python内置求次方函数（源码实现也是快速降幂）
或者位移操作
"""
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number <= 1:
            return number
        else:
            return 2**(number-1) #24ms
            #return 2<<number-2 #22ms
        #**和pow一样，在python源码实现中都是采用的快速降幂，时间复杂度O（log n）
```
##### 题目10：矩形覆盖
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
```python
# -*- coding:utf-8 -*-
"""
此题依然是斐波那契数列的变形

考虑：大矩形设有f(n)种
如果把2*1的小块竖着放在大矩形最左边，那么剩下的有f(n-1)种；
如果把2*1的小块横着放在大矩形左上角，那么剩下的有f(n-2)种（因为左小角一定要放一个）；
所以f(n) =f(n-1)+f(n-2)

利用前面提到的斐波那契公式法，时间复杂度O（log n）
"""
class Solution:
    def rectCover(self, number):
        # write code here
        def fib(n):
            if n < 1:
                return (1, 0)
            f_m_1, f_m = fib(n >> 1)
            if n & 1:
                return f_m_1 * (f_m_1 + 2 * f_m), f_m ** 2 + f_m_1 ** 2
            else:
                return f_m_1 ** 2 + f_m ** 2, f_m * (2 * f_m_1 - f_m)
        if number <=1:
            return number
        else:
            return fib(number)[0]
```

### 《剑指Offer》专题：第一期就到这里啦，感谢您的收藏与转发。
####<center>更多精彩，可访问以下链接<center>

本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork.

github链接：https://github.com/LSayhi
代码仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai

![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-885f1ec27454b67a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)