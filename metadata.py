import glob
import torch
import os
import random
import logging
import re
from torch.utils.data import Dataset
from json_loader import JSONLoader
from data import DATA_SET_DIR
import tqdm
import json
from copy import deepcopy
from collections import Counter
from nltk.tokenize import word_tokenize
from configuration import Configuration
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
LOGGER = logging.getLogger(__name__)
"""
    1.均匀选择在 train和val里的均匀选择 N + K
    2.只在train数据集里选择 N + K
"""
class LWANDataset(Dataset):
    def __init__(self, num_task, p=0.5, N=128, K=32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Configuration['model']['bert'])
        self.N = N
        self.K = K
        self.p = p
        self.num_task = num_task
        self.load_label_descriptions()
        self.train_documents = self.load_dataset('train')
        self.val_documents = self.load_dataset('dev')
        #合并val 和 train
        self.all_samples, self.all_tags, self.train_num = self.process_dataset(self.train_documents)
        self.val_samples, self.val_tags, self.val_num = self.process_dataset(self.val_documents)
        self.all_num = self.train_num + self.val_num
        self.all_num_list = list(range(self.all_num))
        self.train_num_list = list(range(self.train_num))#需为全局变量
        self.all_samples += self.val_samples
        self.all_tags += self.val_tags
        self.num_task_list = list(range(self.num_task))
        self.k = self.num_task * self.p
        self.k = int(self.k)
        self.x = random.sample(self.num_task_list, self.k)

    def __getitem__(self, item):
        for item in self.num_task_list:#遍历所有任务
            if item in self.x:#在x中的选择策略1
                support_data, query_data = self.get_one_task_data_first()
            else:#否则选择策略2
                support_data, query_data = self.get_one_task_data_second()

        return self.data_iter(support_data, query_data)

    def __len__(self):
        return self.num_task


    def load_dataset(self, dataset_name):
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.json'))
        loader = JSONLoader()
        documents = []
        i = 0
        for filename in tqdm.tqdm(sorted(filenames)):
            if i > 159:
                break
            documents.append(loader.read_file(filename))
            i += 1

        return documents  # 返回的是一个列表 元素为document对象

    def process_dataset(self, documents):
        samples = []
        targets = []
        num = 0
        for document in documents:
            samples.append(document.tokens)#样本
            targets.append(document.tags)#标签
            num += 1

        del documents
        return samples, targets, num

    def encode_dataset(self, sequences, tags):
        temp = [' '.join(seq) for seq in sequences]# 32
        samples = self.tokenizer.batch_encode_plus(temp, padding=True, truncation=True, max_length=512,
                                                   return_tensors="pt")
        targets = torch.zeros((len(sequences), len(self.label_ids)), dtype=torch.float32)  # tensor

        for i, (document_tags) in enumerate(tags):
            for tag in document_tags:
                if tag in self.label_ids:
                    targets[i][self.label_ids[tag]] = 1.

        del sequences, tags
        return samples['input_ids'], targets  # 按序排列

    """策略1 在train和valid中随机选择N+K"""
    def get_one_task_data_first(self):
        data = random.sample(self.all_num_list, self.N + self.K)#在46000+6000个随机选一个128+32
        support_data = data[:self.N]
        query_data = data[self.N:]
        return support_data, query_data#[1,2,5,7,8] [0,3,4,6,9]
    """
    策略2 在train中选 原文为先从见过的选标签再根据此选中一个文本 是否与直接从train中选等价 
    可能不等价 原文想不要在一个任务里重复选择一种标签
    样本数量极大 任选未必能使标签重复
    """
    def get_one_task_data_second(self):
        data = random.sample(self.train_num_list,self.N + self.K)
        support_data = data[:self.N]
        query_data = data[self.N:]
        return support_data, query_data

    def data_iter(self,support_data,query_data):
        support_sample = []
        support_target = []
        for i in support_data:
            support_sample.append(self.all_samples[i])
            support_target.append(self.all_tags[i])

        query_sample = []
        query_target = []
        for i in query_data:
            query_sample.append(self.all_samples[i])
            query_target.append(self.all_tags[i])

        support_x, support_y = self.encode_dataset(support_sample, support_target)
        query_x, query_y = self.encode_dataset(query_sample, query_target)
        support_x, support_y, query_x, query_y = support_x.cuda(), support_y.cuda(), query_x.cuda(), query_y.cuda()
        return support_x, support_y, query_x, query_y

    def get_train(self):
        LOGGER.info('fine-tunning')
        LOGGER.info('------------------------------')
        x, y = self.encode_dataset(self.all_samples, self.all_tags)
        x, y = x.cuda(), y.cuda()
        dataset = data.TensorDataset(x, y)
        return data.DataLoader(dataset, 64, shuffle=True)

    def get_test(self):
        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')
        test_documents = self.load_dataset('test')
        test_samples, test_tags, _ = self.process_dataset(test_documents)
        x, y = self.encode_dataset(test_samples, test_tags)
        x, y = x.cuda(), y.cuda()
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, 8, shuffle=True), y

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))#存训练集中所有标签id

        frequent, few = [], []#分别存训练集标签id
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()#dev test中的标签id
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))#dev和test以及train集合中的所有标签与none作差集得到在none中有的但不在数据集里的
        parents = []#获得父母标签
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))#再与所有的父母标签作交集得到 没出现在dev test train集中的标签

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))#出现在test和dev里但是没出现在训练集中的
        true_zero = deepcopy(zero)#浅拷贝一份
        zero = zero + list(none)

        self.label_ids = dict()#4654
        self.margins = [(0, len(frequent) + len(few) + len(true_zero))]#[(0,总长度)]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k + len(group)))#[(0,len(frequent)),(len(frequent),len(frequnet) + len(few)),(len,)]
            for concept in group:
                self.label_ids[concept] = k#frequnt、few、zero从0开始标记 包含了未出现过的母亲节点
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent) + len(few) + len(true_zero))#真正的值

        label_terms = []#[['international', 'affairs'],...]存储解释器里所有的label对应的值
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])


        LOGGER.info('#Labels:         {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

    def calculate_performance(self, model, generator, true_targets):
        pred_tmp = torch.ones((1, 4654))  # 最后需忽略此项
        with torch.no_grad():
            for X, y in generator:
                y = model(X)
                y = y.cpu()
                pred_tmp = torch.cat((pred_tmp, y), dim=0)

        predictions = pred_tmp[1:, :]  # 忽略第一行
        pred = torch.where(predictions > 0.5, 1, 0)
        predictions = predictions.numpy()
        pred_targets = pred.numpy()  # 转numpy
        true_targets = true_targets.cpu()
        true_targets = true_targets.numpy()
        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip(self.margins,
                                                    ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)',
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                r_k = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                p_k = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                rp_k = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                ndcg_k = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('----------------------------------------------------')