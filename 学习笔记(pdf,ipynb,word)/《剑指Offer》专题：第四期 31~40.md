### 《剑指Offer》专题：第四期 31-40
这里是《剑指Offer》专题第四期，本期内容为牛客网序第31-40题的解法与分析。涉及链表、数组、树等数据结构及其相关的算法，题目预览，见下图。
![牛客网31-40.png](https://upload-images.jianshu.io/upload_images/16949178-f515c67db60ebbe2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#### 题目31：整数中1出现的次数
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
```python
# -*- coding:utf-8 -*-
"""
方法1：
    暴力求解：时间O(nlgn),空间O(1)
方法2：
    巧解法：时间O(C*n),C为最大整数的长度（对于Java,C++等语言来说，C为常数）,空间O（1）
方法3：
    规律公式法：时间 O(lgn)，空间O（1），最优解
    参考：https://www.nowcoder.com/questionTerminal/bd7f978302044eee894445e244c7eee6
"""
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        
        #方法1：暴力法：
        """
        def func(num):
            cnt = 0
            while(num):
                if num%10 ==1:
                    cnt +=1
                num = num/10
            return cnt
        count = 0
        for i in range(1,n+1):
            count += func(i)
        return count
        """
        
        #方法2：巧解法，把每个整数转为字符串，统计每个字符串1的个数，再相加
        """
        if n <= 0:#依据题意
            return 0
        count = 0
        for i in range(1,n+1):
            x = str(i)
            for ch in x:
                if ch == '1':
                    count += 1
        return count
        """
        
        #方法3：规律公式求法，最优解
        count = 0
        m = 1
        while m<=n:
            #若是判断包含x(0~9)的个数，只需将（n/m%10 == 1）改成（n/m%10 == x）
            count += (n/m + 8)/10*m + (n/m%10 == 1)*(n%m + 1) 
            m *= 10
        return count
```
#### 题目32：把数组排成最小的数
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
```python
# -*- coding:utf-8 -*-
"""
方法1: 时间复杂度O（n+n-1+n-2+……+2+1）=O（n^2）,空间复杂度O（1）
    要得到最小的数字，其实是要将数组中的数字按构成最小数字的方式排列，然后转换成字符串即可
    而这个方式就是 让数组中数字转成字符串拼接，取两个，若S1+S2>S2+S1则交换S1和S2的位置
方法2：
    方法2思想与方法1一样，只不过是利用python sort函数的cmp参数实现而已
"""
class Solution:
    def PrintMinNumber(self, numbers):
        # 方法1：自己实现排列规则
        if not numbers:
            return ""
        length = len(numbers)
        for i in range(length):
            for j in range(i+1,length):
                a = str(numbers[i])+str(numbers[j])
                b = str(numbers[j])+str(numbers[i])
                if a > b: #前面+后面 大于 后面+前面，则交换数字在数组中位置
                    numbers[i],numbers[j] = numbers[j],numbers[i]
        res = ""
        for num in numbers:
            res += str(num)
        return res
        
        
        #方法2：利用python内置函数简化方法1代码
        """
        def exchangeSignal(x,y):
            if x+y>y+x:
                return 1
            elif x+y<y+x:
                return -1
            else:
                return 0
        if not numbers:
            return ""
        numbers = map(str,numbers)
                #利用cmp参数实现方法1的比较交换功能
        numbers.sort(cmp = lambda x,y:exchangeSignal(x,y))
        res = ""
        for num in numbers:
            res += str(num)
        return res
        """
```
#### 题目33：丑数
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
```python
# -*- coding:utf-8 -*-
"""
根据牛客网代码改编，更加清晰和节省时间
"""
class Solution:
    def GetUglyNumber_Solution(self, index):
        # 鲁棒性代码
        if index < 7: #小于7时，index本身就是第index个丑数
            return index
        p1, p2, p3 = 0, 0, 0 #三个指针位
        uglynum_lst = [1] #初始化
        length_lst = 1  #丑数数组长度计数
        while length_lst< index:
            nextugly = min(2*uglynum_lst[p1],3*uglynum_lst[p2],5*uglynum_lst[p3])
            if uglynum_lst[p1]*2 == nextugly: p1 += 1
            if uglynum_lst[p2]*3 == nextugly: p2 += 1
            if uglynum_lst[p3]*5 == nextugly: p3 += 1
            uglynum_lst.append(nextugly)
            length_lst += 1
        return uglynum_lst[-1] # 返回表尾元素，若需返回前N个丑数，去掉【-1】即可。
"""
参考链接：https://www.nowcoder.com/questionTerminal/6aa9e04fc3794f68acf8778237ba065b
来源：牛客网
通俗易懂的解释：
首先从丑数的定义我们知道，一个丑数的因子只有2,3,5，那么丑数p = 2 ^ x * 3 ^ y * 5 ^ z，换句话说一个丑数一定由另一个丑数乘以2或者乘以3或者乘以5得到，那么我们从1开始乘以2,3,5，就得到2,3,5三个丑数，在从这三个丑数出发乘以2,3,5就得到4，6,10,6，9,15,10,15,25九个丑数，我们发现这种方法会得到重复的丑数，而且我们题目要求第N个丑数，这样的方法得到的丑数也是无序的。那么我们可以维护三个队列：
（1）丑数数组： 1
乘以2的队列：2
乘以3的队列：3
乘以5的队列：5
选择三个队列头最小的数2加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
（2）丑数数组：1,2
乘以2的队列：4
乘以3的队列：3，6
乘以5的队列：5，10
选择三个队列头最小的数3加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
（3）丑数数组：1,2,3
乘以2的队列：4,6
乘以3的队列：6,9
乘以5的队列：5,10,15
选择三个队列头里最小的数4加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
（4）丑数数组：1,2,3,4
乘以2的队列：6，8
乘以3的队列：6,9,12
乘以5的队列：5,10,15,20
选择三个队列头里最小的数5加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
（5）丑数数组：1,2,3,4,5
乘以2的队列：6,8,10，
乘以3的队列：6,9,12,15
乘以5的队列：10,15,20,25
选择三个队列头里最小的数6加入丑数数组，但我们发现，有两个队列头都为6，所以我们弹出两个队列头，同时将12,18,30放入三个队列；
……………………
疑问：
1.为什么分三个队列？
丑数数组里的数一定是有序的，因为我们是从丑数数组里的数乘以2,3,5选出的最小数，一定比以前未乘以2,3,5大，同时对于三个队列内部，按先后顺序乘以2,3,5分别放入，所以同一个队列内部也是有序的；
2.为什么比较三个队列头部最小的数放入丑数数组？
因为三个队列是有序的，所以取出三个头中最小的，等同于找到了三个队列所有数中最小的。
实现思路：
我们没有必要维护三个队列，只需要记录三个指针显示到达哪一步；“|”表示指针,arr表示丑数数组；
（1）1
|2
|3
|5
目前指针指向0,0,0，队列头arr[0] * 2 = 2,  arr[0] * 3 = 3,  arr[0] * 5 = 5
（2）1 2
2 |4
|3 6
|5 10
目前指针指向1,0,0，队列头arr[1] * 2 = 4,  arr[0] * 3 = 3, arr[0] * 5 = 5
（3）1 2 3
2| 4 6
3 |6 9
|5 10 15
目前指针指向1,1,0，队列头arr[1] * 2 = 4,  arr[1] * 3 = 6, arr[0] * 5 = 5
"""
```
####题目34：第一个只出现一次的字符
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
```python
# -*- coding:utf-8 -*-
"""
时间复杂度O（n）
空间复杂度O（1）
map的思想：
    用一个数组记录字符出现的位置，初始化为257，遍历字符串s
    若 该字符的对应数组中的值257，说明未加入数组，置为 i,记录位置
    若 不为257，说明重复出现；置为258
    最后，返回数组中最小的那个数，就是只出现一次字符的位置（<257），否则返回 -1
"""
class Solution:
    def FirstNotRepeatingChar(self, s):
        
        # 鲁棒性检测
        if not s:
            return -1
        
        #用一个数组记录每个字符出现的下标
        record = [257]*256 #设置256为数组长度，hash
        for i in range(len(s)):
            if record[ord(s[i])] == 257:#没出现过，保存出现的位置
                record[ord(s[i])] = i
            else: #出现过了，对应位置置258
                record[ord(s[i])] = 258
        
        #找到下标最小的那个就是第一个只出现一次的字符
        firstOnceindex = 257
        for index in record:
            if index < firstOnceindex:
                firstOnceindex = index
        
        #返回第一个只出现一次字符位置 或者找不到返回-1
        if firstOnceindex<257:
            return firstOnceindex
        else:
            return -1
```
####题目35：数组中的逆序队
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
```python
# -*- coding:utf-8 -*-
"""
此题未能在牛客网上AC，但思路可看。
"""
count = 0
class Solution:
    def InversePairs(self, data):
        global count
        def MergeSort(lists):
            global count
            if len(lists) <= 1:
                return lists
            num = int( len(lists)/2 )
            left = MergeSort(lists[:num])
            right = MergeSort(lists[num:])
            r, l=0, 0
            result=[]
            while l<len(left) and r<len(right):
                if left[l] < right[r]:
                    result.append(left[l])
                    l += 1
                else:
                    result.append(right[r])
                    r += 1
                    count += len(left)-l
            result += right[r:]
            result += left[l:]
            return result
        MergeSort(data)
        return count%1000000007
        
        # 方法1： 两个循环，逐项比较O(n^2)，不通过
        """
        if not data:
            return 0
        n = len(data)
        count = 0
        for i in range(n):
            for j in range(i+1,n):
                if data[i]>data[j]:
                    count = (count+1)
        return count%1000000007
        """
```
####题目36：两个链表的公共结点
输入两个链表，找出它们的第一个公共结点。
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
"""
方法1： 快慢指针，让长的先走几步，然后再一起走，遇到相同结点，返回；时间O（m+n）,空间O（1）
方法2： hash法，把表1hash了，遍历另一个表，查看是否存在表1的结点；时间O（m+n）,空间O（m）
"""
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # 方法1：双指针，快慢指针
        """
        #鲁棒性检测
        if not pHead1 or not pHead2:
            return None
        
        #初始化参数，记录表头结点
        len1,len2 = 1,1
        p1 = ListNode(0)
        p2 = ListNode(0)
        p1.next = pHead1
        p2.next = pHead2
        
        #统计两个表各自长度
        while pHead1 and pHead1.next:
            pHead1 = pHead1.next
            len1 += 1
        while pHead2 and pHead2.next:
            pHead2 = pHead2.next
            len2 += 1
        
        #让长的表先走几步，然后两个表一起走，遇到相同的节点则是公共节点
        if len1>=len2:
            for i in range(len1-len2):
                p1 = p1.next
            while p1!= p2 and p1.next and p2.next:
                p1 = p1.next
                p2 = p2.next
        if len1<len2:
            for i in range(len2-len1):
                p2 = p2.next
            while p1!= p2 and p1.next and p2.next:
                p1 = p1.next
                p2 = p2.next
        if p1 == p2:
            return p1
        else:
            return None
        """
        
        #方法2：hash
        #鲁棒性检测
        if not pHead1 or not pHead2:
            return None
        hashset = set()
        while pHead1:
            hashset.add(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            if pHead2 in hashset:
                return pHead2
            pHead2 = pHead2.next
        return None
```
####题目37：数字在排序数组中出现的次数
统计一个数字在排序数组中出现的次数。
```python
# -*- coding:utf-8 -*-
"""
方法1：先用二分查找找到k,然后分别向左向右遍历，统计 x==k的数字个数
    时间 o(logn)+o(m),n为数组长度，m为数字k出现的次数,如果m很大，时间复杂度就是O(n)了
方法2:优化方法1，我们去二分查找k出现的第一个位置和最后一个位置，最后返回两者差值+1。
    找第一个位置:如果arr[middle]==k,再判断前一位是否!=k,如果是返回middle，否则end=middle-1
    找最后一位，与找第一个位置类似，总的时间复杂度O（logn）+O（logn） = O(logn)
"""
class Solution:
    def GetNumberOfK(self, data, k):
        
        # 方法1：二分查找到k后，再向左向右遍历计数
        """
        #定义二分查找函数
        def binarySearch(arr,begin,end,target):
            while begin <= end:
                middle =(begin+end)//2
                if arr[middle] == target:
                     return middle 
                elif arr[middle]> target:
                     end = middle -1
                else: #arr[middle]< target
                     begin = middle +1
            return -1 #如果没找到，返回-1
        
        #鲁棒性判断
        if not data:return 0
        #得到二分查找的结果，如果-1，返回次数为0；如果非负，向左向右计数
        index = binarySearch(data,0,len(data)-1,k)
        if index == -1:return 0
        count = 0 #要返回的计数值
        i, j = index, index-1
        while i<=len(data)-1:
            if data[i] == k:
                count +=1
            i +=1
        while j>=0:
            if data[j] == k:
                count +=1
            j -=1
        return count
        """
        # 方法2：二分查找第一个k后最后一个k
        def binarySearch_first(arr,begin,end,target):
            while begin <= end:
                middle =(begin+end)//2
                if arr[middle] == target:
                    if (middle>0 and arr[middle-1]!=target) or (middle == 0):
                        #确认middle前面不等于k，或者middle已经在0位置，前面没有数了
                         return middle
                    else:
                        end = middle-1
                elif arr[middle]> target:
                     end = middle -1
                else: #arr[middle]< target
                     begin = middle +1
            return -1 #如果没找到，返回-1
        def binarySearch_last(arr,begin,end,target):
            while begin <= end:
                middle =(begin+end)//2
                if arr[middle] == target:
                    if (middle<end and arr[middle+1]!=target) or (middle == end):
                        #确认middle后面不等于k，或者middle已经在end位置，后面没有数了
                         return middle
                    else:
                        begin = middle+1
                elif arr[middle]> target:
                     end = middle -1
                else: #arr[middle]< target
                     begin = middle +1
            return -1 #如果没找到，返回-1
        if not data:
            return 0
        count = 0
        first = binarySearch_first(data,0,len(data)-1,k)
        last = binarySearch_last(data,0,len(data)-1,k)
        if first == -1 or last == -1:
            return 0
        return last -first +1
```
####题目38：二叉树的深度
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
```python
"""
方法1：DFS 在stack中记录到每个结点的长度，在lst_depth中记录每条路径长度，返回lst_depth的最大值
    时间：O(n),空间O(n)
方法2：DFS 优化方法1，不记录每条路径长度，只记录最大路径长度 更加节省空间
方法3：BFS 实现方法1
方法4：BFS 实现方法2
"""
class Solution:
    def TreeDepth(self, pRoot):
        # 方法1：DFS 记录每条路径长度，然后返回最大值
        """
        if not pRoot:#鲁棒性检测
            return 0
        stack = [[pRoot,1]]#结点和到此结点的路径长度
        lst_depth = [] #存放每个路径的长度
        while stack:
            node,depth = stack.pop()
            if node.right:
                stack.append([node.right,depth+1])
            if node.left:
                stack.append([node.left,depth+1])
            if not node.right and not node.left:
                lst_depth.append(depth)
        return max(lst_depth)
        """
        #方法2：DFS 不记录每条路径长度，只记录当前最大路径长度，节省空间
        """
        if not pRoot:#鲁棒性检测
            return 0
        stack = [[pRoot,1]]#结点和到此结点的路径长度
        lst_depth = 1 #存放每个路径的长度
        while stack:
            node,depth = stack.pop()
            if node.right:
                stack.append([node.right,depth+1])
            if node.left:
                stack.append([node.left,depth+1])
            if not node.right and not node.left:
                lst_depth = max(lst_depth,depth)
        return lst_depth
        """
        #方法3：BFS 对应方法1的版本，就不写啦，因为要写方法4
        #方法4：BFS 对应方法2的版本，这里写一下
        if not pRoot:#鲁棒性检测
            return 0
        queue = [[pRoot,1]]#结点和到此结点的路径长度
        lst_depth = 1 #存放每个路径的长度
        while queue:
            node,depth = queue.pop(0)
            if node.right:
                queue.append([node.right,depth+1])
            if node.left:
                queue.append([node.left,depth+1])
            if not node.right and not node.left:
                lst_depth = max(lst_depth,depth)
        return lst_depth
```
####题目39：平衡二叉树
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
```python
# -*- coding:utf-8 -*-
"""
深度优先遍历二叉树
    某个结点的子树不平衡或者自己不平衡则返回-1
    其他情况返回树深（从叶向上数）
    最后调用DFS函数查看返回值是否为-1，不为-1，则返回True,否则返回False
"""
class Solution:
    def IsBalanced_Solution(self, pRoot):
        #方法1： 深度优先遍历结点，结点不平衡则返回-1
        def dfs(root):
            #这是给叶子结点的
            if root is None:return 0 
            l_depth  = dfs(root.left)
            r_depth = dfs(root.right)
            #左子树不平衡，或右子树不平衡，或当前树结点不平衡 直接返回-1
            if l_depth == -1 or r_depth == -1 or abs(l_depth - r_depth)>1:
                return -1
            return 1 + max(l_depth, r_depth) # 返回当前树的深度
        
        if not pRoot:return True#鲁棒性代码
        return dfs(pRoot) != -1 # 成立说明平衡
```
####题目40：数组中只出现一次的数字
一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
```python
# -*- coding:utf-8 -*-
"""
方法1：O(n),O(n)
    用一个字典记录每个数字出现的次数，然后遍历找出次数等于1的数字
方法2：O(n),o(1)
    我们知道 两个相同的字符相异或为0,0和其它字符异或等于该字符，且异或满足交换律
    因此，我们把整个数组异或，会得到两个单次字符异或的结果，这个结果二进制中至少有一个1
    找到结果中1出现的任何一个位置，按此位是否为1，将数组可分为两组，一组为0，一组为1
    那么两个单次字符必然分别出现两组中，而两组中其它数字都是重复出现的
    因此，对这两组中的字符异或，得到的两个结果就是 这两个出现一次的数字
"""
class Solution:
    def FindNumsAppearOnce(self, array):
        #方法2：异或
        #函数，判断num中的idx位(从低向高)是否为1
        def Bit_equalone(num, idx):
            num = num >> idx
            return num & 1
        #鲁棒性检测
        if not array: return [0,0]
        #数组array直接异或，得到两个单次字符异或的结果
        temp = 0
        for ch in array:
            temp ^= ch
        #找到结果中，1出现的任意一个位置（我们取最低位）
        index = 0
        while (temp&1) == 0:
            index += 1
            temp = temp >> 1
        #将数组array按index位是否为1分为两组，分别异或
        res1, res2 = 0, 0
        for ch in array:
            if Bit_equalone(ch,index):
                res1 ^= ch
            else:
                res2 ^= ch
        return [res1,res2]#返回两个只出现一次的数
        #方法1：字典
        """
        if not array: return 0,0
        dic = {}
        res = [0]*2
        for ch in array:
            dic[ch] = dic.get(ch,0)+1
        i=0
        for ch,count in dic.items():
            if count == 1:
                res[i]=ch
                i += 1
        return res[0],res[1]
        """
```
### 第四期内容就到这啦~
本文章所有代码及文档均以上传至github中，感谢您的star,fork and rp.

github链接：https://github.com/LSayhi
仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai
![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-14dffa49f6812dd2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
