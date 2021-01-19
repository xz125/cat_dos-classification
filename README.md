# cat_dos-classification


＃cat_dos分类＃

本文是一个二分类，你也可以将它扩展为多分类，此处目的在于学习Pytorch流程

首先下载文件后

有一个data 文件 这里需要放置你的数据

data 文件如下：

-data --train --类别1 --类别2 .... --test --类别1 --类别2 ....

log 文件 是放置 可视化tensorboard 文件

output 文件是放置权重的文件

test 放置测试文件

如果你的数据跟data 一样情况下，可以不用使用make_file.py 文件 如果你的类别数据混乱在一起，请参考make_file.py 文件
2.train.py 是训练文件

3.test.py 是测试文件
