### 六张图带你了解 Redis 底层数据结构

大家好久不见！最近忙于写论文，有几期没更新了，趁今天无心学习，总结了一下 Redis的 底层数据结构，画了几张图与大家分享一下~

**1.简单动态字符串**

Redis没有直接C语言的传统字符串表示，而是构建了简单动态字符串来作为默认的字符串表示，其相比C字符串，具有许多优点，见下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619153757481.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)



**2.链表**

Reids自己构建了链表的实现，应用与列表键、发布与订阅、慢查询、监视器等场合
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619153846375.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)


**3.字典**

Redis构建了自己的字典实现，应用相当广泛，比如 Redis的数据库就是以字典作为底层实现，字典还是哈希键的底层实现之一,Redis中字典的实现如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619153911677.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)
  **4.跳跃表**

跳跃表是一种有序的数据结构，跳跃表支持平均O（logN）,最坏O（N）的节点查找，效率与平衡二叉树媲美，实现又较为简单~跳跃表节点的level可以视作是索引，高层索引数目少于低层，因此查找等操作时间复杂度可类比于二分查找~
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619153936214.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)
**5.整数集合**

整数集合是集合键的底层实现之一，当一个集合包含整数元素，且数量不多时，Redis就会使用整数集合作为集合键的底层实现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619153956927.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)

  **6.压缩列表**

压缩列表是列表键和哈希键的底层呢过实现之一，当列表键包含少量列表项，并且每个列表项为小整数值或长度比较短的字符串时，Redis使用压缩列表实现列表键。其优点是节省内存~

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190619154007440.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xTYXloaQ==,size_16,color_FFFFFF,t_70)

好啦，本期内容就到这里

####  <center style="margin: 0px; padding: 0px; font-size: inherit; color: inherit; line-height: inherit;">更多精彩，可访问往期文章</center>



有任何疑问，欢迎后台留言，如果对文章内容有更深或独特的见解，欢迎投稿或者到github中PR。本文章所有代码及文档均以上传至github中，感谢您的rp,star and fork。

github链接：https://github.com/LSayhi

CSDN链接：https://blog.csdn.net/LSayhi

![image](https://upload-images.jianshu.io/upload_images/16949178-9b624d27d51edf78?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





