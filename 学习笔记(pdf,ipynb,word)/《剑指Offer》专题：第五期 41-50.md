### 《剑指Offer》专题: 第五期 41-50
这里是《剑指Offer》专题系列第五期，一共十题，每题对应多种解法，一起来看看吧。
###正文
#### 题目41：和为S的连续正数序列
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
输出描述:
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
```python
"""
方法1:利用数学方法 时间复杂度O（sqrt(n)）
方法2：滑动窗口法，时间复杂度O（sqrt(n)）
"""
class Solution:
    def FindContinuousSequence(self, tsum):
        #方法1：数学规律
        """
        # 鲁棒性代码
        if tsum <=0: return []
        res = []
        temp = []
        x = int((2*tsum)**0.5)# 等差数列求和公式得出 n<sqrt(2*sum)
        for n in range(x,1,-1): #n为序列的长度，范围为了减少n扫描的次数
            #n为偶数和n为奇数时，满足以下条件才能组成连续和S
            if tsum%n==0.5*n or ((n%2) and (tsum%n==0)): 
                for i in range(tsum/n-(n-1)/2,tsum/n+n/2+1):
                    temp.append(i)
                res.append(temp)
                temp = []
        return res
        """
        
        #方法2： 滑动窗口法（不定长,可伸缩）
        l, r, sum, res = 1, 2, 3, []
        while l<r and l<2*tsum**0.5:# l<2*tsum**0.5是从方法1偷师来的，加上会减少时间复杂度
            if sum>tsum:#如果大了，则把最前面的从窗口中去掉
                sum -= l
                l += 1 #左边标志右移
            else:
                if sum==tsum:#相等则加入返回列表
                    res.append([i for i in range(l,r+1)])
                r += 1 #窗口右移
                sum += r
        return res
```
####题目42：和为S的两个数字
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
输出描述:
对应每个测试案例，输出两个数，小的先输出
```python
# -*- coding:utf-8 -*-
"""
方法1： 时间复杂度o(n)，空间复杂度o(1)
    利用数学知识：我们知道两个数的和固定时，两个数越接近，则乘积越大，反之，乘积越小
    于是，我们定义两个指针small,large，一个从第0位开始递增，一个从最后一位开始递减
    当满足两个位置的数字之和等于tsum，则跳出循环，并返回这两个位置的值，此时乘积一定是最小的。
    如果当small>=large时，退出循环，此时未找到两个数之和等于tsum，返回空数组。
"""
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if not array:return []
        small, large = 0, len(array)-1
        result =[]
        while small<large: #
            if array[small]+array[large]== tsum:
                result = [array[small],array[large]]
                break
            elif array[small]+array[large]<tsum:
                small +=1
            else:# array[small]+array[large]>tsum:
                large -=1
        return result
```
####题目43：左旋转字符串
汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
```python
# -*- coding:utf-8 -*-
"""
方法1：时间复杂度O（len(s)）,空间复杂度O（len(s)）
    秀是秀，可是没啥意义啊...面试会跪的
方法2：时间复杂度O(len(s))
    (X+Y) = （X(翻转)+Y(翻转))(翻转)
"""
class Solution:
    def LeftRotateString(self, s, n):
        #方法2 (X+Y) = （X(翻转)+Y(翻转))(翻转)
        def reverse(arr):
            i, j = 0, len(arr)-1
            while i<j:
                arr[i],arr[j] =arr[j],arr[i]
                i += 1
                j -= 1
            return arr
        if not s:return ""
        slist = [ch for ch in s] #python字符串是不可修改的，所以转成数组列表
        num = n%(len(slist))
        res = reverse( reverse(slist[:num]) + reverse(slist[num:]) )
        return ''.join(res) #转回字符串
        # 方法1：取余后切片拼接
        """
        if not s: return ""
        num = n%len(s) #取余，因为n可能会大于s的长度
        res = s[num:]+s[:num]
        return res #原字符串没变，只是新建了一个字符串
        """
```
####题目44：翻转单词顺序列
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
```python
# -*- coding:utf-8 -*-
"""
方法1：将字符串转为列表，再逆向存储每个单词，再转回字符串
"""
class Solution:
    def ReverseSentence(self, s):
        # 方法1：先去掉空格，翻转、在加入空格
        if not s:return ""
        snew = s.split(' ') #去掉空格，转为列表
        n = len(snew)-1
        res = []
        for i in range(n,-1,-1):#反向存储到res中
            res.append(snew[i])
        return ' '.join(res)#变成字符串，加空格

```
####题目45：扑克牌顺子
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
```python
# -*- coding:utf-8 -*-
"""
方法1: 时间复杂度O(n),空间复杂度O(1)
    1.连续序列，最大值-最小值 = length -1 ,但因为有0（任意数），所以，最大值-最小值<=length-1
    2.如果出现重复数字，其满足条件1，但不能构成连续序列，用定长映射数组记录每个数字出现的次数
    由此，遍历一遍给出的所有数字，求出最大值最小值，求出长度
    如果满足条件1，并且每个数字出现的次数不大于1，则可构成连续序列
"""
class Solution:
    def IsContinuous(self, numbers):
        # 方法1
        if not numbers: return False #鲁棒性代码
        czero, cnotzero = 0, 0
        smaller, larger = 15, -1  
        flag = [0]*13 #数组存储每个数字出现的次数（A-K，1-13）
        for num in numbers:
            if num == 0:#等于0时，零的个数加1
                czero += 1
            else:
                cnotzero += 1 #非0的个数加1
                flag[num] += 1 # 给非0数字的次数+1
                if flag[num]>1: #出现重复非0数字，返回False
                    return False
                if num > larger: #求出非0最大值
                    larger = num
                if num < smaller:#求出非0最小值
                    smaller = num
                    
        #连续序列，最大值与最小值之差等于length-1,但这有0当任意数，所以<=length-1都行
        if larger-smaller<= czero+cnotzero-1:
            return True
        else:
            return False
                                               
```
####题目46：孩子们的游戏
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
```python
# -*- coding:utf-8 -*-
"""
方法1：直接利用递推公式的 时间复杂度O（n）,空间复杂度O(1)

这是一个典型的约瑟夫环问题，问题分析如下:
长度为n（0~n-1）报数为m的序列s，其出局的下标为(m-1)%n,下标m%n为新的报数起点，且序列长度-1，m不变。
这就构成了一个递归的求解过程，一直递归下去我们能找到剩余的一位幸运儿；
但是，如何根据下一层（长度n-1）的报数点，获得其在上一层(长度为n)的位置呢？（反过来求解）
编号转换的例子（左边长度为n的序列报数的下标，右边长度n-1序列报数的下标）：
k     <--> 0  
k+1   <--> 1
k+2   <--> 2
...
k-2   <--> n-2
我们发现 index[n] = (index[n-1]+m)%n
于是有递推公式 f[1]=0;f[i]=(f[i-1]+m)%i (i>=2)，当i=n时，得到结果.
"""
class Solution:
    def LastRemaining_Solution(self, n, m):
        # 方法1：先找规律，再逆向
        if n==0 or m ==0:return -1
        index = 0
        for i in range(2,n+1):
            index = (index + m) % i
        return index
```
####题目47：求1+2+3+……+n
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
```python
# -*- coding:utf-8 -*-
"""
由于不能使用if,while等，因此得使用递归
我们知道n==0时，递归结束  但是不能用if
所以，我们可以用 and 来实现当n ==0时，不执行递归
注：A and B ,如果A为0，则不执行
如果A和B不为0，表达式的值是A、B中大的那个，例如 2 and 1,值为2
"""
class Solution:
    def Sum_Solution(self, n):
        # write code here
        return n and (n + self.Sum_Solution(n-1))
```
####题目48：不用加减乘除做加法
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
```python
# -*- coding:utf-8 -*-
"""
三步走：
1.计算不进位和a；
2.计算进位b
3.计算a+b，如果有进位，回到1；如果没有进位，则返回a+b的不进位和
注：python 与C++、Java区别
"""
class Solution:
    def Add(self, num1, num2):
        # C++或JAVA实现
        """
        while num2!=0: #直到进位为0
            Sum = num1^num2 #异或，不进位加法
            carry = (num1 & num2)<<1 # 进位
            num1 = Sum 
            num2 = carry  
        return num1
        """
        # write code here
        # 由于题目要求不能使用四则运算，那么就需要考虑使用位运算
        # 两个数相加可以看成两个数的每个位先相加，但不进位，然后在加上进位的数值
        # 如12+8可以看成1+0=1 2+8=0，由于2+8有进位，所以结果就是10+10=20
        # 二进制中可以表示为1000+1100 先每个位置相加不进位，
        # 则0+0=0 0+1=1 1+0=1 1+1=0这个就是按位异或运算
        # 对于1+1出现进位，我们可以使用按位与运算然后在将结果左移一位
        # 最后将上面两步的结果相加，相加的时候依然要考虑进位的情况，直到不产生进位
        # 注意python没有无符号右移操作，所以需要越界检查
        while num2:
            result = (num1 ^ num2) & 0xffffffff
            carry = ((num1 & num2) << 1) & 0xffffffff
            num1 = result
            num2 = carry
        if num1 <= 0x7fffffff:
            result = num1
        else:
            result = ~(num1^0xffffffff)
        return result
```
####题目49：把字符串转换成整数
将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。
输入描述:
输入一个字符串,包括数字字母符号,可以为空
输出描述:
如果是合法的数值表达则返回该数字，否则返回0
```python
# -*- coding:utf-8 -*-
"""
方法1：利用了int()
方法2：满足题意
"""
class Solution:
    def StrToInt(self, s):
        # 方法1  利用了int()
        """
        if not s:return 0
        arr = ['0','1','2','3','4','5','6','7','8','9']
        res = ''
        flag = 1
        if s[0] == '+' and len(s)>1:
            s =s[1:]
            flag = 1
        if s[0] == '-' and len(s)>1:
            s = s[1:]
            flag = -1
        for ch in s:
            if ch in arr:
                res += ch
            else:
                return 0
        ret = 0
        for i in range(len(res)):
            ret += int(res[i])*10**(len(res)-i-1)
        return ret*flag
        """
        #方法2 
        numlist=['0','1','2','3','4','5','6','7','8','9','+','-']
        sum=0
        label=1#正负数标记
        if s=='':
            return 0
        for string in s:
            if string in numlist:#如果是合法字符
                if string=='+':
                    label=1
                    continue
                if string=='-':
                    label=-1
                    continue
                else:
                    sum=sum*10+numlist.index(string)
            if string not in numlist:#非合法字符
                sum=0
                break#跳出循环
        return sum*label
```
####题目50：数组中重复的数字
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
```python
"""
方法1：时间复杂度o(n),空间复杂度O（1）
    遍历数组，
    如果数组i位置的值等于i,则什么也不做，遍历下一个元素；
    如果，数字i位置不等于i,则判断i位置的值x是否等于x位置的值；
    如果arr[i] == arr[x],重复了，返回True
    如果arr[i] != arr[x],交换两个位置的值，让x处于arr[x]位置
方法2：hash 时间O（n）,空间O（n），较为简单，省略
"""
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        if not numbers:return False
        for i in range(len(numbers)-1):
            if numbers[i]!= i:
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0] = numbers[i]
                    return True
                else:
                    temp = numbers[i]
                    numbers[i] = numbers[temp]
                    numbers[temp] = temp
        return False
```
### 《剑指Offer》第五期：就到这里啦，感谢您的收藏与转发。
####<center>更多精彩，可访问往期文章<center>
有任何疑问，欢迎后台留言，如果对文章内容有更深或独特的见解，欢迎投稿或者到github中PR。本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork.

github链接：https://github.com/LSayhi
代码仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai

![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-885f1ec27454b67a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)