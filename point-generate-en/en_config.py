
# stanford_corenlp 所在的路径，绝对路径
stanford_corenlp = "D:\stanford-corenlp-full-2018-10-05"

expect_train_file_num = 287227
expect_test_file_num = 11490
expect_val_file_num = 13368

# 正常情况下，一个文件对应一个样本，但是有些文件中没有文章，或者没有摘要，就去掉了

expect_train_sample_num = 287113
expect_test_sample_num = 11490
expect_val_sample_num = 13368




SPECIAL_TOKEN = ['<pad>','<unk>','<start>','<stop>']


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


hidden_dim = 256
embed_dim = 128