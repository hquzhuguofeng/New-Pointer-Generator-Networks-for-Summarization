# --token_data
# F:\data\en\tokenized
# --use_coverage
# --pointer_gen
# --do_train
# --do_decode

import en_config as config
import argparse
import logging
import torch
from torch.optim import Adagrad,Adam
import os
import warnings
from data_util.vocab import Vocab
from data_util.data import get_features
from train_util import train
from decoder import decoder
from model.model import PointerGeneratorNetworks
import random
import code
import numpy as np

logger = logging.getLogger(__name__)

def check(args,vocab):

    train_token_dir = os.path.join(args.token_data,"train")
    test_token_dir = os.path.join(args.token_data,"test")
    val_token_dir = os.path.join(args.token_data,"val")

    assert os.path.exists(train_token_dir)
    assert os.path.exists(test_token_dir)
    assert os.path.exists(val_token_dir)

    # 检查这几个文件夹的文件数目是否和配置中的对应
    trian_token_file_num = len(os.listdir(train_token_dir))
    if trian_token_file_num != config.expect_train_file_num:
        warnings.warn("实际训练样本文件数量:{}与预期:{}不一致".format(trian_token_file_num, config.expect_train_num))

    test_token_file_num = len(os.listdir(test_token_dir))
    if test_token_file_num != config.expect_test_file_num:
        warnings.warn("实际测试样本文件数量:{}与预期:{}不一致".format(test_token_file_num, config.expect_test_num))

    val_token_file_num = len(os.listdir(val_token_dir))
    if val_token_file_num != config.expect_val_file_num:
        warnings.warn("实际验证样本文件数量:{}与预期:{}不一致".format(val_token_file_num, config.expect_val_num))

    # 创建文件夹：features_50000_400_100
    # 500000是词表的大小，400文章最大长度，摘要最大长度
    feature_dir = "{}_{}_{}_{}".format(args.feature_dir_prefix,args.vocab_num,args.article_max_len,args.abstract_max_len)
    args.feature_dir = os.path.join(".",feature_dir)

    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
        print("创建的特征目录:{}".format(feature_dir))

    # 在特征文件夹内创建：train/ test / val 的子文件夹
    train_feature_dir = os.path.join(".",feature_dir,"train")
    test_feature_dir = os.path.join(".", feature_dir, "test")
    val_feature_dir = os.path.join(".", feature_dir, "val")
    if not os.path.exists(train_feature_dir):
        os.mkdir(train_feature_dir)
    if not os.path.exists(test_feature_dir):
        os.mkdir(test_feature_dir)
    if not os.path.exists(val_feature_dir):
        os.mkdir(val_feature_dir)

    # 对应有正文和摘要的 例子留下来，分别没有正文和摘要的段落不要了！
    # 总的文件个数 / 每个特征文件装个的文件个数 = 一共有多少个特征文件
    except_train_feature_file_num = config.expect_train_sample_num // args.example_num
    
    if 0 != trian_token_file_num % args.example_num:
        except_train_feature_file_num += 1
    real_train_feature_file_num = len(os.listdir(train_feature_dir))
    if real_train_feature_file_num == 0:
        get_features(token_dir=train_token_dir,feature_dir=train_feature_dir,vocab = vocab,args = args,data_set="train")
    elif real_train_feature_file_num != except_train_feature_file_num:
        raise ValueError("train feature dir {} not empty".format(train_feature_dir))

    except_test_feature_file_num = config.expect_test_sample_num // args.example_num
    if 0 != test_token_file_num % args.example_num:
        except_test_feature_file_num += 1
    real_test_feature_file_num = len(os.listdir(test_feature_dir))
    if real_test_feature_file_num == 0:
        get_features(token_dir=test_token_dir, feature_dir=test_feature_dir,vocab=vocab ,args=args, data_set="test")
    elif real_test_feature_file_num != except_test_feature_file_num:
        raise ValueError("test feature dir {} not empty".format(test_feature_dir))

    except_val_feature_file_num = config.expect_val_sample_num // args.example_num
    if 0 != val_token_file_num % args.example_num:
        except_val_feature_file_num += 1
    real_val_feature_file_num = len(os.listdir(val_feature_dir))
    if real_val_feature_file_num == 0:
        get_features(token_dir=val_token_dir, feature_dir=val_feature_dir, vocab=vocab, args=args, data_set="val")
    elif real_val_feature_file_num != except_val_feature_file_num:
        raise ValueError("val feature dir {} not empty".format(val_feature_dir))
    # real_train_feature_file_num == 36
    # real_test_feature_file_num == 2
    # real_val_feature_file_num == 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--token_data",default="E:\\0000_python\\point-genge\\point-generate\\en\\cnn-dalymail-tokenizer\\tokenized",type = str,required=True,
                       help="包含 train，test，evl 和 vocab.json的文件夹")

    parser.add_argument("--feature_dir_prefix",default="features",
                        help="train，test，evl从样本转化成特征所存储的文件夹前缀置")

    parser.add_argument("--do_train",action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_decode", action='store_true',
                        help="是否对测试集进行测试")

    parser.add_argument("--example_num", default= 1024 * 8,type = int,
                        help="每一个特征文件所包含的样本数量")

    parser.add_argument("--article_max_len",default=400,type=int,
                       help="文章的所允许的最大长度")

    parser.add_argument("--abstract_max_len", default=100, type=int,
                        help="摘要所允许的最大长度")

    parser.add_argument("--vocab_num",default=50000,type = int,
                        help="词表所允许的最大长度")

    parser.add_argument("--pointer_gen",action='store_true',
                        help="是否使用指针机制")

    parser.add_argument("--use_coverage",action="store_true",
                        help="是否使用汇聚机制")

    parser.add_argument("--no_cuda", action='store_true',
                        help="当GPU可用时，选择不用GPU")

    parser.add_argument("--epoch_num", default=10,type = int,
                        help="epoch")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="train batch size")

    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="evaluate batch size")

    parser.add_argument("--hidden_dim", default=256,type=int,
                        help="hidden dimension")
    parser.add_argument("--embedding_dim",default=128,type=int,
                        help="embedding dimension")
    parser.add_argument("--coverage_loss_weight",default=1.0,type=float,
                        help="coverage loss weight ")
    parser.add_argument("--eps",default=1e-12,type = float,
                        help="log(v + eps) Avoid  v == 0,")
    parser.add_argument("--dropout",default= 0.5,type =float,
                        help="dropout")

    parser.add_argument("--lr",default=1e-3,type=float,
                        help="learning rate")
    parser.add_argument("--max_grad_norm",default=1.0,type=float,
                        help="Max gradient norm.")

    parser.add_argument("--adagrad_init_acc", default=0.1, type=float,
                        help="learning rate")

    parser.add_argument("--adam_epsilon",default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--gradient_accumulation_steps",default=1,type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",default="output",type=str,
                        help="Folder to store models and results")

    parser.add_argument("--evaluation_steps",default = 500,type=int,
                        help="Evaluation every N steps of training")
    parser.add_argument("--seed",default=4321,type=int,
                        help="Random seed")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args.seed)

    vocab_file = os.path.join(args.token_data, 'vocab.json') # 加载词表文件
    assert os.path.exists(vocab_file)
    vocab = Vocab(vocab_file=vocab_file, vob_num=args.vocab_num) # 构建词表 word_to_id  id_to_word 统计词的个数，将pad/unk/start/stop加入词库

    check(args, vocab=vocab)
    

    model = PointerGeneratorNetworks(vob_size=args.vocab_num,embed_dim=args.embedding_dim,hidden_dim=args.hidden_dim,
                                     pad_idx = vocab.pad_idx,dropout=args.dropout,pointer_gen = args.pointer_gen,
                                     use_coverage=args.use_coverage)

    model = model.to(args.device)
    model = model.to(args.device)

    if args.do_train:
        optimizer = Adam(model.parameters(),lr = args.lr)
        # train(args = args,model=model,optimizer = optimizer,with_eval = True)
    if args.do_decode:
        decoder(args,model,vocab=vocab)

if __name__ == "__main__":
    main()




