import en_config as config
import torch
import os
import logging
import code


class Feature(object):
    def __init__(self,article,abstract,unique_name,encoder_input,decoder_input,decoder_target,encoder_input_with_oov,oovs,
                 decoder_target_with_oov,max_encoder_len,max_decoder_len,pad_idx = 0):

        assert len(decoder_input) == len(decoder_target)
        self.article = article
        self.abstract = abstract

        self.unique_name = unique_name
        self.encoder_input,self.encoder_mask = self._add_pad_and_gene_mask(encoder_input,max_encoder_len,pad_idx)
        self.encoder_input_with_oov = self._add_pad_and_gene_mask(encoder_input_with_oov, max_encoder_len, pad_idx,
                                                                  return_mask=False)

        self.decoder_input,self.decoder_mask = self._add_pad_and_gene_mask(decoder_input,max_decoder_len,pad_idx)

        self.decoder_target = self._add_pad_and_gene_mask(decoder_target,max_decoder_len,pad_idx,return_mask=False)
        self.decoder_target_with_oov = self._add_pad_and_gene_mask(decoder_target_with_oov, max_decoder_len, pad_idx,
                                                                   return_mask=False)
        self.oovs = oovs
        self.oov_len = len(oovs)

    @classmethod
    def _add_pad_and_gene_mask(cls,x,max_len,pad_idx = 0,return_mask = True):
        pad_len = max_len - len(x)
        assert pad_len >= 0

        if return_mask:
            mask = [1] * len(x)
            mask.extend([0] * pad_len)
            assert len(mask) == max_len

        x.extend([pad_idx] * pad_len)
        assert len(x) == max_len

        if return_mask:
            return x,mask
        else:
            return x


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line:
      return line
  if line=="":
      return line
  if line[-1] in config.END_TOKENS:
      return line
  return line + " ."


def article_word_to_idx_with_oov(article_list,vocab):
    indexes = []
    oovs = []
    for word in article_list:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word not in oovs:
                oovs.append(word)
            oov_idx = oovs.index(word)
            indexes.append(vocab.get_vob_size() + oov_idx)
        else:
            indexes.append(idx)
    return indexes,oovs


def abstract_target_idx_with_oov(abstract_list , vocab , oovs):
    target_with_oov = []
    for word in abstract_list[1:]:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word in oovs:
                target_with_oov.append(vocab.vob_num+oovs.index(word))
            else:
                target_with_oov.append(vocab.unk_idx)
        else:
            target_with_oov.append(idx)
    return target_with_oov


def read_example_convert_to_feature(example_path_name,article_len,abstract_len,vocab,index,point = True):

    article = []
    abstract = []
    flag = False
    with open(example_path_name,'r',encoding="utf-8") as f:
        for line in f: # 一行行读取
            # print("before_line:{}\n",line)

            # 小写化，在句尾加上空格+句号 ' .'
            line = fix_missing_period(line.strip().lower())
            # print("after_line:{}\n",line)
            # input(">>>")
            # 
            # 并将正文和摘要分开来
            if "" == line:
                continue
            elif line.startswith("@highlight"):
                flag = True
            elif flag:
                abstract.extend(line.split(' '))
                flag = False
            else:
                article.extend(line.split(' '))
    f.close()
    
    # 有一些没有摘要，有一些没有文章，都不要了
    if 0 == len(article) or 0 == len(abstract):
        return None
    unique_name = example_path_name.split("\\")[-1].split(".")[0]
    print_idx = 20
    if index < print_idx:
        print("====================================={}=====================================".format(unique_name))


    if index < print_idx:
        print("原始文章长度[{}]===[{}]".format(len(article)," ".join(article)))
        print("原始摘要长度[{}]===[{}]".format(len(abstract)," ".join(abstract)))
    
    # 截断,限制文章的长度
    article = article[:article_len] # 原文长度840 ，截断400

    if index < print_idx:
        print("截断后的文章长度[{}]===[{}]".format(len(article)," ".join(article)))
    article_indexes = [vocab.word_2_idx(word) for word in article] # 将正文转化为向量

    

    #加上 start 和 end
    abstract = [vocab.start_token] + abstract + [vocab.stop_token]
    # 截断，限制摘要的长度
    abstract = abstract[:abstract_len+1] # 摘要加上start 和 end token之后再截断

    if index < print_idx:
        print("截断后的摘要长度[{}]===[{}]".format(len(abstract)," ".join(abstract)))



    abstract_indexes = [vocab.word_2_idx(word) for word in abstract] # 将摘要转化为向量

    decoder_input = abstract_indexes[:-1] # 构建输入的摘要和目标摘要，语言模型
    decoder_target = abstract_indexes[1:]

    assert len(decoder_input) == len(decoder_target)

    if point:
        # 更新正文和摘要的词表，
        encoder_input_with_oov,oovs = article_word_to_idx_with_oov(article,vocab)
        decoder_target_with_oov = abstract_target_idx_with_oov(abstract,vocab,oovs)


    feature_obj = Feature(article = article,
                          abstract = abstract[1:],
                          unique_name = unique_name,
                          encoder_input =article_indexes,
                          decoder_input = decoder_input,
                          decoder_target = decoder_target,
                          encoder_input_with_oov = encoder_input_with_oov,
                          oovs = oovs,
                          decoder_target_with_oov = decoder_target_with_oov,
                          max_encoder_len = article_len,
                          max_decoder_len = abstract_len,
                          pad_idx = vocab.pad_idx)
    # code.interact(local = locals())


    if index < print_idx:

        print("encoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input])))
        print("encoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_mask])))
        print("encoder_input_with_oov :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input_with_oov])))
        print("decoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_input])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_mask])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target_with_oov])))
        print("oovs          :[{}]".format(" ".join(oovs)))
        print("\n")

    return feature_obj


def get_features(token_dir,feature_dir,vocab,args,data_set = "train",example_num = 1024*8):
    assert os.path.exists(token_dir)
    assert os.path.exists(feature_dir)
    assert data_set in ["train","test","val"]
    assert 0 == example_num % 1024 and example_num > 1024


    token_file_list = os.listdir(token_dir)
    sample_num = len(token_file_list) # 287227

    feature_file_idx = 0
    feature_file_prefix = "{}".format(data_set) # 'train'
    features = []

    
    for idx,token_file in enumerate(token_file_list):
        if example_num == len(features):
            feature_file_idx += 1
            feature_file_name = "{}_{:0>2d}".format(feature_file_prefix, feature_file_idx)
            feature_file_name_path = os.path.join(feature_dir, feature_file_name)
            torch.save(features, feature_file_name_path)
            print("本次转换完成{},已经转换完成{}个，一共{}个,占比{:.2%}  存储特征文件{}".
                  format(len(features), idx, sample_num, float(idx) / sample_num, feature_file_name))
            features = []

        # 一个故事的绝对路径 e:/../xxx/train/xxxx.story
        file_name = os.path.join(token_dir,token_file)
        
        feature_obj = read_example_convert_to_feature(example_path_name=file_name,
                                                      article_len=args.article_max_len,
                                                      abstract_len = args.abstract_max_len,
                                                      index=idx,
                                                      vocab = vocab)
        # code.interact(local = locals()) 
        if feature_obj is not None:
            features.append(feature_obj)

    if 0 != len(features):
        feature_file_idx += 1
        feature_file_name = "{}_{:0>2d}".format(feature_file_prefix,feature_file_idx)
        feature_file_name_path = os.path.join(feature_dir, feature_file_name)
        torch.save(features, feature_file_name_path)

        print("本次转换完成{},已经转换完成{}个，一共{}个,占比{:.2%}  存储特征文件{}".
              format(len(features),idx, sample_num,float(idx+1) / sample_num,feature_file_name))
        features = []