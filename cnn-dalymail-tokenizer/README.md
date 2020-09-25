# cnn-dalymail-tokenizer

## 作用：对数据集cnn和dalymail进行分词，并构建词表

url_lists中的文件每一行一个链接对应一个文件，链接的sha1码就是文件的名字。



cnn 和 dalymail [源数据链接](https://cs.nyu.edu/~kcho/DMQA/)

国内的百度网盘我打包了一份，与源数据相同  [链接](https://pan.baidu.com/s/107EoRnytcHGA2JUmoUGRfQ ) 提取码:k8v3



分词需要用到stanford-corenlp，[stanford-corenlp下载链接](https://stanfordnlp.github.io/CoreNLP/download.html)

国内的有一个已经下载好的版本，在百度网盘[链接](https://pan.baidu.com/s/1jT7bufghBIjJY89DquAdcg ) 提取码:pnzk

stanford-corenlp 依赖jar

分词过程中，如果遇到编码问题，不能够被分，忽略掉，没什么影响。例如：
Untokenizable: ? (U+202C, decimal: 8236)
Untokenizable: ? (U+202A, decimal: 8234)
Untokenizable: ? (U+202A, decimal: 8234)
Untokenizable: ? (U+202C, decimal: 8236)


还有一个与项目本身关系不大的坑，使用pycharm写代码，如果把源文件，或者分词后的文件放在项目目录下，pycharm会尝试去读取这些文本文件，产生的一个后果就是电脑非常的卡顿。


如果不想分词，百度网盘我也上传了一个已经分词完成的cnn和dalymail，包括train，test，val文件和vocab.json词频文件。[链接](https://pan.baidu.com/s/1P3wARiRpyW5bHW7ZVLPnPA ) 提取码:7e0j


url_lists:链接链接：https://pan.baidu.com/s/1Z4O044sv-WjyotEqJw0dVA  提取码：3zpq
