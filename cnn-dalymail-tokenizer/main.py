# --original_data
# F:\data\en\cnn-dailymail
# --output_dir
# F:\data\en\\tokenized


import os
import cnn_daily_config as config
import warnings
import hashlib
import subprocess
import datetime
import argparse
import collections
import json
import code


# 记录数据的绝对路径
# 创建映射数据表从原始路径到处理完之后存放的位置
class Tokenizer(object):
    def __init__(self,data_dir,output_dir):
        cnn_stories = os.path.join(data_dir,"cnn","stories")
        daily_mail = os.path.join(data_dir,"dailymail","stories")
        # code.interact(local = locals())

        assert  os.path.exists(cnn_stories)
        assert  os.path.exists(daily_mail)

        self.cnn_stories = cnn_stories
        self.daily_mail = daily_mail

        train_map_file = os.path.join(".", "train_map.txt") # 创建映射数据表从原始路径到处理完之后存放的位置
        test_map_file = os.path.join(".", "test_map.txt")
        val_map_file = os.path.join(".", "val_map.txt")
        self.train_map_file = train_map_file
        self.test_map_file = test_map_file
        self.val_map_file = val_map_file

        self.output_dir = os.path.join(output_dir) # ./tokenized 处理完之后存放的目录地址
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.train_dir = os.path.join(output_dir,"train") # ./tokenized/train
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if 0 != len(os.listdir(self.train_dir)):
            raise ValueError("生成分词后的train set文件夹不为空")


        self.test_dir = os.path.join(output_dir,"test") # ./tokenized/test
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if 0 != len(os.listdir(self.test_dir)):
            raise ValueError("生成分词后的test set文件夹不为空")

        self.val_dir = os.path.join(output_dir,"val") # ./tokenized/val
        if not os.path.exists(self.val_dir):
            os.mkdir(self.val_dir)
        if 0 != len(os.listdir(self.val_dir)):
            raise ValueError("生成分词后的 val set文件夹不为空")
        # code.interact(local = locals())

    def gene_map_file(self):
        print("gene_map_file\n")
        # 创建UTL连接，里面都是网站链接
        # http://web.archive.org/web/20150401100102id_/http://www.cnn.com/2015/04/01/europe/france-germanwings-plane-crash-main/
        train_url_file = os.path.join(".","url_lists","all_train.txt") # '.\\url_lists\\all_train.txt'
        test_url_file = os.path.join(".", "url_lists", "all_test.txt")
        val_url_file = os.path.join(".", "url_lists", "all_val.txt")
        
        # 统计cnn和dailymail里面story的个数是否和预期的一样
        cnn_num = len(os.listdir(self.cnn_stories))
        if config.expected_cnn_num != cnn_num:
            warnings.warn("预期的CNN_stories数量是{},而实际的数量{}".format(config.expected_cnn_num, cnn_num))
        print("done 1\n")
        # code.interact(local = locals())
        daily_mail_num = len(os.listdir(self.daily_mail))
        if config.expected_daily_mail_num != daily_mail_num:
            warnings.warn("预期的daily_mail_stories数量是{},而实际的数量{}".format(config.expected_daily_mail_num, daily_mail_num))
        print("done 2\n")

        def url_to_sha1(url):
            h = hashlib.sha1()
            h.update(url.encode())
            return h.hexdigest()

        def get_sha1_from_url_file(url_file):
            with open(url_file,"r",encoding='utf') as f:
                sha_list = [url_to_sha1(line.strip()) for line in f]
                f.close()
            return sha_list

        # 统计file表的元素集合
        train_set = set(get_sha1_from_url_file(train_url_file))
        test_set = set(get_sha1_from_url_file(test_url_file))
        val_set = set(get_sha1_from_url_file(val_url_file))

        train_map_str_list = []
        test_map_str_list = []
        val_map_str_list = []


        print("开始生成train，test，val新旧文件路径的映射文件")

        # self.cnn_stories='E:\\python_data\\cnn-dailymail\\cnn\\stories'
        # self.daily_mail='E:\\python_data\\cnn-dailymail\\dailymail\\stories'
        old_dir_list = [self.cnn_stories,self.daily_mail]
        for d in old_dir_list:
            file_list = os.listdir(d) # 就是cnn/story/的每个故事的列表
            for file_name in file_list: # 获得到每个故事的文件名
                unique_name = file_name.split(".")[0] # 获取到每个故事文件名去除.story

                # old_path_name=E:\\python_data\\cnn-dailymail\\cnn\\stories\000c835555db62e319854d9f8912061cdca1893e.story
                old_path_name = os.path.join(d,file_name) 
                # 判断文件实在那个集合中
                # train_map_str_list的每个元素是[初始数据存放的地方, 现在存放的地方]
                # 'E:\\python_data\\cnn-dailymail\\cnn\\stories\\0001d1afc246a7964130f43ae940af6bc6c57f01.story\t./tokenized\\train\\0001d1afc246a7964130f43ae940af6bc6c57f01.story'
                if unique_name in train_set:
                    new_path_name = os.path.join(self.train_dir,file_name)
                    train_map_str_list.append("{}\t{}".format(old_path_name,new_path_name))
                elif unique_name in test_set:
                    new_path_name = os.path.join(self.test_dir, file_name)
                    test_map_str_list.append("{}\t{}".format(old_path_name,new_path_name))
                elif unique_name in val_set:
                    new_path_name = os.path.join(self.val_dir, file_name)
                    val_map_str_list.append("{}\t{}".format(old_path_name, new_path_name))
                else:
                    warnings.warn("文件[{}]不属于train，test，val任何一个数据集".
                                  format(old_path_name))
        # 生成 数据原来存放地方 以及tokenizer之后数据存放的地方
        with open(self.train_map_file,"w",encoding="utf-8") as f:
            f.write("\n".join(train_map_str_list))
            f.close()
        print("train set 新旧文件路径的映射文件生成完成")
        # code.interact(local = locals())

        with open(self.test_map_file,"w",encoding="utf-8") as f:
            f.write("\n".join(test_map_str_list))
            f.close()
        print("test set 新旧文件路径的映射文件生成完成")


        with open(self.val_map_file,"w",encoding="utf-8") as f:
            f.write("\n".join(val_map_str_list))
            f.close()
        print("val set 新旧文件路径的映射文件生成完成")
        


    def tokenize(self,map_file):
        assert os.path.exists(map_file)

        cmd = ["java","-cp", config.stanford_nlp_jar_path, config.tokenizer, "-ioFileList", "-preserveLines", map_file]
        subprocess.call(cmd)


    def tokenize_all(self):

        self.gene_map_file()

        print("*********************************************Train*********************************************")
        print("Train set 分词开始，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.tokenize(self.train_map_file)
        print("Train set 分词结束，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        # code.interact(local = locals())

        print("*********************************************Test*********************************************")
        print("Test set 分词开始，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.tokenize(self.test_map_file)
        print("Test set 分词结束，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        print("*********************************************Val*********************************************")
        print("Val set 分词开始，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.tokenize(self.val_map_file)
        print("Val set 分词结束，时间：{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


    def gene_vocab_file(self,vocab_file_name):

        vocab_file = os.path.join(self.output_dir, vocab_file_name) # 词表文件
        word_freq = collections.Counter() # 计数器
        print("*********************************************{}*********************************************".
              format(vocab_file))
        # code.interact(local = locals())
        file_list = os.listdir(self.train_dir) # 获取到分词完之后训练数据文件列表
        for file in file_list: # 获取到每个文件名
            path_file = os.path.join(self.train_dir,file) # 将上面的文件名补充成绝对路径
            with open(path_file,'r',encoding='utf-8') as f: # 分别打开这些文件
                for line in f: # 一行行读取
                    line = line.strip() # 忽略空白符，并跳过 摘要
                    # code.interact(local = locals())
                    if "" == line:
                        continue
                    if line.startswith("@highlight"):
                        continue
                    line = line.lower()
                    line_word = line.split(" ")
                    line_word = [word.strip() for word in line_word]
                    line_word = [word for word in line_word if "" != word]
                    word_freq.update(line_word)
                f.close()

        word_freq = word_freq.most_common(len(word_freq))
        word_freq = dict(word_freq)

        with open(vocab_file,"w",encoding='utf-8') as f:
            json.dump(word_freq,f)
            f.close()




# "F:\Python_code\data\cnn-dailymail"


def main():

    parser = argparse.ArgumentParser()

    # 例如，我的路径就是F:\data\en\cnn-dailymail,这个路径下，有以下内容
    # F:\data\en\cnn-dailymail\cnn\stories
    # F:\data\en\cnn-dailymail\dailymail\stories
    #
    parser.add_argument("--original_data",default=None,type = str,required=True,
                       help="包含 /cnn/stories，/dailymail/stories 的文件夹")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="分词后文件所存储的文件夹")

    parser.add_argument("--word_freq",default="vocab.json",type = str,
                        help="词表文件")

    args = parser.parse_args()

    tokenize = Tokenizer(data_dir = args.original_data,output_dir = args.output_dir)

    tokenize.tokenize_all()
    tokenize.gene_vocab_file(vocab_file_name=args.word_freq)




if __name__ == "__main__":
    main()







