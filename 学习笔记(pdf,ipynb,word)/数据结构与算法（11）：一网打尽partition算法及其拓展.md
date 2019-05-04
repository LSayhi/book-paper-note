### 数据结构与算法（11）: 一网打尽partition算法及其拓展
Partition（划分）算法在快速排序、TopK问题、三色排序等问题上都能展现其巨大价值，本次文章讲述的是parition算法在这些算法问题上的应用以及partition算法是如何实现的，文章包括以下6个算法的实现：

- 1.partition算法的单指针实现；
- 2.partition算法的双指针实现；
- 3.快速排序 基于双指针partition的实现
- 4.寻找第K个最小的数
- 5.寻找前K个最小的数
- 6.荷兰三色旗问题
#### 算法的IO解释：
- 1.输入一个数组A，把数组A按枢轴（pivot）划分为两部分，左边小于等于枢轴，右边大于等于枢轴，枢轴在两者中间，但是左右两边不排序，并且返回枢轴的位置(下标).
例如：[1,2,6,3,4,5] pivot =4 得到 [2,1,3,4,6,5] 下标：3
- 2.与1一样，不过采用双指针扫描，时间复杂度更低。
- 3.基于2实现快速排序
- 4.输入一个数组，输出第K个最小的数，并对数组排序了（A[k-1]为第K个最小的数）
- 5.输入一个数组，输出前K个最小的数，并对数组排序了（同上）
- 6.输入一个数组，输出分界点pos_small和pos_big,pos_small左边全是小于target的值，pos_big右边全是大于target的值，两者之间的值等于target,并且数组被排序了。荷兰问题是6的子集中，即输入的数组值只有三种情况，代码完全不变。
##### 算法的实现如下，解释见注释
```python
# -*- coding: utf-8 -*-
# @time :  2019/5/3  22:21
# @Author: LSayhi
# @github: https://github.com/LSayhi

"""1.一个指针实现划分函数"""
def partition_OnePoint(array,begin,end):
    pivot = array[begin]#枢轴，可以取数组中任意一个位置的值，这里取第一个，方便说明
    pos = begin #记录分界点（其左边<=pivot，其右边>pivot）

    for i in range(begin+1,end):#从左到右遍历数组
        if array[i] <= pivot:#如果i处的值小于枢轴，则把pos向后移移一位
            pos += 1
            if pos != i:# 如果pos<i了（说明pos位置的值大于pivot），则把pos的值与i处的交换
                temp = array[i]
                array[i] = array[pos]
                array[pos] = temp

    array[begin] = array[pos]
    array[pos] = pivot  #最后两行是交换pos处与pivot（begin处的值），把pivot放入中间，左边全部<=pivot,右边全部>pivot
    return pos

"""2.两个指针实现划分函数"""
def partition_TwoPoint(array,begin,end):
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

"""3.快速排序 双指针版本"""
def quickSort(array,begin,end):
    if begin>=end :#鲁棒性检测及递归推出
        return
    pos = partition_TwoPoint(array,begin,end)#找到划分点
    quickSort(array,begin,pos)#递归划分左边
    quickSort(array, pos + 1, end)#递归划分右边

"""4.寻找第K个小的数字"""
def kth_Number(array,k):
    begin, end = 0, len(array)-1
    if not array or k<=0 or k > end +1:#鲁棒性检测
        return 0
    kth_num = 0
    while begin< end:
        pos = partition_TwoPoint(array,begin,end)#找到划分点
        if pos == k -1:#刚好是第K小的，则返回
            kth_num = array[pos]
            break
        elif pos > k -1:#说明第k小的在左边
            end = pos
        else:#说明第k小的在右边
            begin = pos + 1
    return kth_num

"""5.前K个小的数字"""
def k_Number(array,k):
    begin, end = 0, len(array)-1
    if not array or k<=0 or k > end +1:
        return []
    k_numlst = []
    while begin< end:
        pos = partition_TwoPoint(array,begin,end)
        if pos == k -1:
            k_numlst = array[0:pos+1] #返回前k个数，其它与kth_number一样
            break
        elif pos > k -1:
            end = pos
        else:
            begin = pos + 1
    return k_numlst

"""6.三路划分，荷兰三色旗问题"""
#根据partition算法改进为三路划分，左边小于target;右边大于target;中间等于target
def threeway_partition(array,target):
    pos_small, pos_big = 0, len(array)-1
    pos = 0
    while pos <= pos_big:
        if array[pos] < target:
            array[pos],array[pos_small] = array[pos_small],array[pos] #swap
            pos += 1
            pos_small += 1
        elif array[pos] > target:
            array[pos],array[pos_big] = array[pos_big],array[pos] #swap
            pos_big -= 1
        else:
            pos += 1
    return pos_small, pos_big+1

def main():#测试代码
    A = [5,9,2,1,4,7,8,3,6]
    B = [5,9,2,1,4,7,8,3,6]
    C = [5,9,2,1,4,7,8,3,6]
    D = [5,9,2,1,4,7,8,3,6]
    E = [5,9,2,1,4,7,8,3,6]
    F = [0,1,2,1,2,0,1,3,5]
    G = [0,2,2,1,1,0,0,2,1]

    a = partition_OnePoint(A,0,len(A)-1)
    b = partition_TwoPoint(B,0,len(B)-1)
    quickSort(C,0,len(C)-1)
    d = kth_Number(D,6)
    e = k_Number(E,6)
    f = threeway_partition(F,2)
    g = threeway_partition(G,1)

    print("A after partition_onepoint: " + "divide index =" + str(a), ", array =" + str(A))
    print("B after partition_twopoint: " + "divide index =" + str(b), ", array =" + str(B))
    print("C after quicsort: " + str(C))
    print("Found the kth_number in array D is "+ str(d))
    print("Found smallest k numbers in array E is " + str(e))
    print("F sorted:"+str(F)+",divide indexs:"+str(f)+",three parts:"+ str(F[:f[0]])+str(F[f[0]:f[1]])+str(F[f[1]:]))
    print("G sorted:"+str(G)+",three parts:"+str(G[:g[0]])+str(G[g[0]:g[1]])+str(G[g[1]:])+",Dutch Flag problem solved!")

if __name__ == "__main__":
    main()
```
测试代码运行结果：
![测试结果](https://upload-images.jianshu.io/upload_images/16949178-5cd59c0edca7987d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考文献:
https://www.cnblogs.com/zuilehongdou/p/6197716.html
https://www.jianshu.com/p/583ae17759a8
### 数据结构与算法（11）：就到这里啦，感谢您的收藏与转发。
####<center>更多精彩，可访问往期文章<center>
有任何疑问，欢迎后台留言，如果对文章内容有更深或独特的见解，欢迎#投稿#或者到github中PR。本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork.

github链接：https://github.com/LSayhi
代码仓库链接：https://github.com/LSayhi/Algorithms

CSDN链接：https://blog.csdn.net/LSayhi

微信公众号：AI有点可ai

![AI有点可ai.jpg](https://upload-images.jianshu.io/upload_images/16949178-885f1ec27454b67a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)