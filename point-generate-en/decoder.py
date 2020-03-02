import os
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from rouge import Rouge
import json
import code



def get_features_from_cache(cache_file):

    features = torch.load(cache_file)

    all_encoder_input = torch.tensor([f.encoder_input for f in features], dtype=torch.long)
    all_encoder_mask = torch.tensor([f.encoder_mask for f in features],dtype = torch.long)

    all_decoder_input = torch.tensor([f.decoder_input for f in features],dtype=torch.long)
    all_decoder_mask = torch.tensor([f.decoder_mask for f in features],dtype=torch.int)

    all_decoder_target = torch.tensor([f.decoder_target for f in features],dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor([f.encoder_input_with_oov for f in features],dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor([f.decoder_target_with_oov for f in features],dtype=torch.long )
    all_oov_len = torch.tensor([f.oov_len for f in features],dtype=torch.int)

    dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                            all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len)

    unique_names = [f.unique_name for f in features]
    abstracts = [f.abstract for f in features]
    oovs = [f.oovs for f in features]
    return dataset,unique_names,oovs,abstracts


def from_feature_get_model_input(features,hidden_dim,device = torch.device("cpu"),pointer_gen = True,
                                 use_coverage = True):

    # all_encoder_input=[1,400]     all_encoder_mask=[1,400]   all_decoder_input=[1,100]   all_decoder_mask=[1,100]
    # all_decoder_target=[1,400]    all_encoder_input_with_oov=[1,400] all_decoder_target_with_oov=[1,100]
    # all_oov_len [3]
    all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len = features

    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if pointer_gen:                # 当时用指针网络时，decoder_target应该要带上oovs
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:                # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
            oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32) # [1,3]
    else:                                  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
        all_encoder_input_with_oov = None


    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(),dtype=torch.float32)          # 注意数据格式是float

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)   # 注意数据格式是float

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage]
    model_input = [t.to(device) if t is not None else None for t in model_input]

    return model_input


def idx_to_token(idx,oov_word,vocab):

    if idx < vocab.vob_num:
        return vocab.idx_2_word(idx)
    else:
        idx = idx - vocab.vob_num
        return oov_word[idx]




def decoder(args,model,vocab):
    output_dir = args.output_dir
    model_dir_list = os.listdir(output_dir)

    # model_dir_list 将文件加上路径
    model_dir_list = [os.path.join(".",output_dir,model_dir) for model_dir in model_dir_list]
    
    # 判断文件是否为文件夹，排除掉是log文件
    model_dir_list = [model_dir for model_dir in model_dir_list if os.path.isdir(model_dir)]

    # 前面训练了几百轮的模型，效果估计没有后面的好，所以倒序一下，
    model_dir_list = list(reversed(model_dir_list)) # ['.\\output\\step_00500', '.\\output\\step_00001']

    test_feature_dir = os.path.join(args.feature_dir, "test") # .\\features_50000_400_100\\test
    feature_file_list = os.listdir(test_feature_dir) # ['test_01','test_02']

    rouge = Rouge()

    model_iterator = trange(int(len(model_dir_list)), desc = "Model.bin File")
    
    for model_idx in model_iterator:
        model_dir = model_dir_list[model_idx] # 取出第i个文件夹 step_00500

        decoder_dir = model_dir
        predict_file = os.path.join(decoder_dir,"predict.txt") # E:\0000_python\point-genge\point-generate\en\point-generate-en\output\step_00500\predict.txt
        score_json = {}
        score_json_file = os.path.join(decoder_dir,"score.json") # 目前还没有
        result_json = {}
        result_json_file = os.path.join(decoder_dir,"result.json")

        model_path_name = os.path.join(model_dir,"model.bin") # ./output/model.bin
        model.load_state_dict(torch.load(model_path_name)) # 加载这个模型的参数
        model = model.to(args.device) # 配置模型在设备上
        model.eval() # 配置模型进入eval()

        file_iterator = trange(int(len(feature_file_list)), desc=decoder_dir) # test文件夹下数目的迭代器

        for file_idx in file_iterator:
            file = feature_file_list[file_idx] # 拿到的是test的数据： tokenizered/test/test_01
            path_file = os.path.join(test_feature_dir,file) # 绝对路径

            # test_dataset 含8个单元，test数据集含有8192不同的文件，oovs也有8192个单词，abstracts也有8192个
            test_dataset,unique_names,oovs,abstracts = get_features_from_cache(path_file)
            
            # 将上面的数据整理成一个个的迭代器
            test_sampler = SequentialSampler(test_dataset)
            train_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

            data_iterator = tqdm(train_dataloader, desc=decoder_dir)
            for i, batch in enumerate(data_iterator):
                
                batch = from_feature_get_model_input(batch, hidden_dim=args.hidden_dim, device=args.device,
                                                     pointer_gen=args.pointer_gen, use_coverage=args.use_coverage)
                current_unique_name = unique_names[i] # 当前处理的这个文件名
                current_oovs = oovs[i] # 当前这篇文章oov单词
                current_abs = abstracts[i][:-1]        # 去掉stop  当前摘要
                # 进入模型的decode阶段：
                beam = model(encoder_input = batch[0],
                             encoder_mask= batch[1],
                             encoder_with_oov = batch[2],
                             oovs_zero = batch[3],
                             context_vec = batch[4],
                             coverage = batch[5],
                             mode = "decode",
                             beam_size = 10
                             )
                
                # 去除 start token
                hypothesis_idx_list = beam.tokens[1:]
                if vocab.stop_idx == hypothesis_idx_list[-1]:
                    hypothesis_idx_list = hypothesis_idx_list[:-1]


                hypothesis_token_list = [idx_to_token(index,oov_word = current_oovs,vocab = vocab)
                                         for index in hypothesis_idx_list]

                hypothesis_str = " ".join(hypothesis_token_list)
                reference_str = " ".join(current_abs)

                result_str = "{}\t{}\t{}\n".format(current_unique_name,reference_str,hypothesis_str)
                with open(file=predict_file,mode='a',encoding='utf-8') as f:
                    f.write(result_str)
                    f.close()
                rouge_score = rouge.get_scores(hyps = hypothesis_str,refs= reference_str)
                score_json[current_unique_name] = rouge_score[0]
                
        with open(score_json_file, 'w') as f:
            json.dump(score_json,f)
            f.close()



        rouge_1_f = []
        rouge_1_p = []
        rouge_1_r = []
        rouge_2_f = []
        rouge_2_p = []
        rouge_2_r = []
        rouge_l_f = []
        rouge_l_p = []
        rouge_l_r = []


        for name,score in score_json.items():
            rouge_1_f.append(score["rouge-1"]['f'])
            rouge_1_p.append(score["rouge-1"]['p'])
            rouge_1_r.append(score["rouge-1"]['r'])
            rouge_2_f.append(score["rouge-2"]['f'])
            rouge_2_p.append(score["rouge-2"]['p'])
            rouge_2_r.append(score["rouge-2"]['r'])
            rouge_l_f.append(score["rouge-l"]['f'])
            rouge_l_p.append(score["rouge-l"]['p'])
            rouge_l_r.append(score["rouge-l"]['r'])

        mean_1_f = sum(rouge_1_f) / len(rouge_1_f)
        mean_1_p = sum(rouge_1_p) / len(rouge_1_p)
        mean_1_r = sum(rouge_1_r) / len(rouge_1_r)
        mean_2_f = sum(rouge_2_f) / len(rouge_2_f)
        mean_2_p = sum(rouge_2_p) / len(rouge_2_p)
        mean_2_r = sum(rouge_2_r) / len(rouge_2_r)
        mean_l_f = sum(rouge_l_f) / len(rouge_l_f)
        mean_l_p = sum(rouge_l_p) / len(rouge_l_p)
        mean_l_r = sum(rouge_l_r) / len(rouge_l_r)



        result_json['mean_1_f'] = mean_1_f
        result_json['mean_1_p'] = mean_1_p
        result_json['mean_1_r'] = mean_1_r
        result_json['mean_2_f'] = mean_2_f
        result_json['mean_2_p'] = mean_2_p
        result_json['mean_2_r'] = mean_2_r
        result_json['mean_l_f'] = mean_l_f
        result_json['mean_l_p'] = mean_l_p
        result_json['mean_l_r'] = mean_l_r
        with open(result_json_file, 'w') as f:  # test.json文本，只能写入状态 如果没有就创建
            json.dump(result_json, f)  # data转换为json数据格式并写入文件
            f.close()



