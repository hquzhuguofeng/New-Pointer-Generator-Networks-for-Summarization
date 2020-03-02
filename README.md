# en 
指针生成网络，英文数据集下生成摘要

### cnn-dalymail-tokenizer
cnn和dalymail数据集的分词代码


### point-generate-en
指针生成网络在cnn和dalymail数据集下的应用


## 运行
先是cnn-dailymail-tokenizer
python main.py --original_data E:\python_data\cnn-dailymail --output_dir ./tokenized
E:\python_data\cnn-dailymail是我存放cnn-dailymail的地方
这步需要挺多时间的。

然后进入point-generate-en
python main.py --token_data xxx\tokenized --use_coverage --pointer_gen --do_train --do_decode
xxx_toenized 是存放分词后的文件夹






