# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import sharedmem
import numpy as np
import os.path
import gc
import json
from argparse import Namespace
import time
from progressbar import progressbar
import pandas as pd
import gc
import unittest
from collections import defaultdict
FOLDDATA_WRITE_VERSION = 4

def _add_zero_to_vector(vector):
  return np.concatenate([np.zeros(1, dtype=vector.dtype), vector])

def get_dataset_from_json_info(
                dataset_name,
                info_path,
                store_pickle_after_read = True,
                read_from_pickle = True,
                feature_normalization = True,
                purge_test_set = True,
                shared_resource = False,
                relvance_strategy=None):
  with open(info_path) as f:
    all_info = json.load(f)
  assert dataset_name in all_info, 'Dataset: %s not found in info file: %s' % (dataset_name, all_info.keys())

  set_info = all_info[dataset_name]
  assert set_info['num_folds'] == len(set_info['fold_paths']), 'Missing fold paths for %s' % dataset_name
#   print(all_info)
  if feature_normalization:
    num_feat = set_info['num_unique_feat']
  else:
    num_feat = set_info['num_nonzero_feat']
  if "feature_filter_dim" in set_info.keys():
    feature_filter_dim=set_info["feature_filter_dim"]
  else:
    feature_filter_dim=[]
  return DataSet(dataset_name,
                 set_info['fold_paths'],
                 set_info['num_relevance_labels'],
                 num_feat,
                 set_info['num_nonzero_feat'],
                 already_normalized=set_info['query_normalized'],
                 feature_filter_dim=feature_filter_dim
                )

class DataSet(object):

  """
  Class designed to manage meta-data for datasets.
  """
  def __init__(self,
               name,
               data_paths,
               num_rel_labels,
               num_features,
               num_nonzero_feat,
               store_pickle_after_read = True,
               read_from_pickle = True,
               feature_normalization = True,
               purge_test_set = True,
               shared_resource = False,
               already_normalized=False,
              feature_filter_dim=[]):
    self.name = name
    self.feature_filter_dim=feature_filter_dim
    self.num_rel_labels = num_rel_labels
    self.num_features = num_features
    self.data_paths = data_paths
    self.store_pickle_after_read = store_pickle_after_read
    self.read_from_pickle = read_from_pickle
    self.feature_normalization = feature_normalization
    self.purge_test_set = purge_test_set
    self.shared_resource = shared_resource
    self._num_nonzero_feat = num_nonzero_feat
    self._num_nonzero_feat = num_nonzero_feat
  def num_folds(self):
    return len(self.data_paths)
    
  def get_data_folds(self):
    return [DataFold(self, i, path) for i, path in enumerate(self.data_paths)]
class decoms:
  def __init__(self):
    self.decomp=None


class DataFoldSplit(object):
  def __init__(self, datafold, name, doclist_ranges, feature_matrix, label_vector,queryLeastLength=0,rankListLength=5, relvance_strategy=None):
    self.datafold = datafold
    self.name = name
    self.doclist_ranges = doclist_ranges
    self.feature_matrix = feature_matrix
    self.label_vector = label_vector
    self.queriesList=self.filtered_query_sizes(queryLeastLength)
    self.exposure=np.zeros_like(label_vector).astype(np.float64)
    self.query_freq=np.zeros(self.num_queries_orig())
    self.cacheLists=defaultdict(list)
    self.docFreq=np.zeros_like(label_vector)
    self.weightedClicksAver=np.zeros_like(label_vector).astype(np.float64)
    self.ClickSum=np.zeros_like(label_vector).astype(np.float64)
    self.relvance_strategy=relvance_strategy
    self.decomps=defaultdict(decoms)
    self.PLFairModel=None
    self.n_doc=label_vector.shape[0]
    self.rankListLength=rankListLength
    self.ItemFreqEachRank=np.zeros(shape=(self.n_doc,rankListLength)).astype(np.float64)
  def set_rankListLength(self,rankListLength):
    self.rankListLength=rankListLength
    self.ItemFreqEachRank=np.zeros(shape=(self.n_doc,rankListLength)).astype(np.float64)
  def set_relvance_strategy(self,relvance_strategy):
    self.relvance_strategy=relvance_strategy
  def voidFeature(self):
    self.feature_matrix=np.zeros(1)
  def updateStatistics(self,qid,clicks,ranking,positionBias):
    self.query_freq[qid]+=1
    q_docFreq=self.query_values_from_vector(qid,self.docFreq)
    q_ItemFreqEachRank=self.query_values_from_vector(qid,self.ItemFreqEachRank)
    exposure=self.query_values_from_vector(qid,self.exposure)
    ClickSum=self.query_values_from_vector(qid,self.ClickSum)
    weightedClicksAver=self.query_values_from_vector(qid,self.weightedClicksAver)
    indArange=np.arange(ranking.shape[0])
    np.add.at(q_ItemFreqEachRank,[ranking,indArange],1)
    np.add.at(q_docFreq,ranking,1)
    np.add.at(exposure,ranking,positionBias)
    np.add.at(ClickSum,ranking,clicks)
    # np.add.at(self.weightClicksSum,ranking,clicks/positionBias)
    weightedClicksAver[:]=ClickSum/(np.clip(exposure,1e-5,np.inf))
  def getEstimatedAverageRelevance(self,userFeature=None):
      if self.relvance_strategy=="TrueAverage":
          return self.label_vector
      elif self.relvance_strategy=="EstimatedAverage":
          return self.weightedClicksAver
      else:
          raise 
  def num_queries_orig(self):
    return self.doclist_ranges.shape[0] -1
  def num_queries(self):
    return len(self.queriesList)
  def num_docs(self):
    return self.label_vector.shape[0]

  def query_values_from_vector(self, qid, vector):
    s_i, e_i = self.query_range(qid)
    return vector[s_i:e_i]
 
  def query_range(self, query_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    return s_i, e_i
  def get_subset_doc_ids(self,q_ids):
      doc_ids=[]
      for query_id in q_ids:
          s_i, e_i=self.query_range(query_id)
          doc_ids.append(list(range(s_i,e_i)))
      doc_ids=np.concatenate(doc_ids)
      return doc_ids
  def query_size(self, query_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    return e_i - s_i
  def global2local(self,q_ids,global_ids):
        q_ids=np.array(q_ids)
        s_is=self.doclist_ranges[q_ids]
        local_ids=global_ids-s_is[:,None]
        return local_ids

  def query_sizes(self):
    return (self.doclist_ranges[1:] - self.doclist_ranges[:-1])

  def filtered_query_sizes(self,queryLeastLength,queryMaximumLength=np.inf):
    selected_query=np.where(np.logical_and(self.query_sizes()>queryLeastLength , self.query_sizes()<queryMaximumLength) )[0]
    self.queriesList=selected_query
    print("number of query in data split",len(selected_query),flush=True)
    return self.queriesList

  def max_query_size(self):
    return np.amax(self.query_sizes())

  def query_labels(self, query_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    return self.label_vector[s_i:e_i]

  def query_feat(self, query_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    return self.feature_matrix[s_i:e_i, :]

  def doc_feat(self, query_index, doc_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    assert s_i + doc_index < self.doclist_ranges[query_index+1]
    return self.feature_matrix[s_i + doc_index, :]

  def doc_str(self, query_index, doc_index):
    doc_feat = self.doc_feat(query_index, doc_index)
    feat_i = np.where(doc_feat)[0]
    doc_str = ''
    for f_i in feat_i:
      doc_str += '%s:%f ' % (self.datafold.feature_map[f_i], doc_feat[f_i])
    return doc_str


class DataFold(object):

  def __init__(self, dataset, fold_num, data_path):
    self.name = dataset.name
    self.num_rel_labels = dataset.num_rel_labels
    self.num_features = dataset.num_features
    self.fold_num = fold_num
    self.data_path = data_path
    self._data_ready = False
    self.store_pickle_after_read = dataset.store_pickle_after_read
    self.read_from_pickle = dataset.read_from_pickle
    self.feature_normalization = dataset.feature_normalization
    self.purge_test_set = dataset.purge_test_set
    self.shared_resource = dataset.shared_resource
    self._num_nonzero_feat = dataset._num_nonzero_feat
    self.feature_filter_dim=dataset.feature_filter_dim
  def max_query_size(self):
    return np.amax((
        self.train.max_query_size(),
        self.validation.max_query_size(),
        self.test.max_query_size(),
      ),)

  def data_ready(self):
    return self._data_ready

  def clean_data(self):
    del self.train
    del self.validation
    del self.test
    self._data_ready = False
    gc.collect()

  def _make_shared(self, numpy_matrix):
    """
    Avoids the copying of Read-Only shared memory.
    """
    if self._data_args.n_processing == 1:
      return numpy_matrix
    if numpy_matrix is None:
      return None
    shared = sharedmem.empty(numpy_matrix.shape, dtype=numpy_matrix.dtype)
    shared[:] = numpy_matrix[:]
    return shared

  def _read_file(self, path, feat_map, purge):
    '''
    Read letor file.
    '''
    queries = []
    cur_docs = []
    cur_labels = []
    current_qid = None
    print("preprocessing ",path)
    with open(path,"r") as f:
      content = f.readlines()
    for line in progressbar(content):
      info = line[:line.find('#')].split()
      qid = info[1].split(':')[1]
      label = int(info[0])
      feat_pairs = info[2:]

      if current_qid is None:
        current_qid = qid
      elif current_qid != qid:
        stacked_documents = np.stack(cur_docs, axis=0)
        if self.feature_normalization:
          stacked_documents -= np.amin(stacked_documents, axis=0)[None, :]
          safe_max = np.amax(stacked_documents, axis=0)
          safe_max[safe_max == 0] = 1.
          stacked_documents /= safe_max[None, :]

        np_labels = np.array(cur_labels, dtype=np.int64)
        if not purge or np.any(np.greater(np_labels, 0)):
          queries.append(
            {
              'qid': current_qid,
              'n_docs': stacked_documents.shape[0],
              'labels': np_labels,
              'documents': stacked_documents
            }
          )
        current_qid = qid
        cur_docs = []
        cur_labels = []

      doc_feat = np.zeros(self._num_nonzero_feat)
      for pair in feat_pairs:
        feat_id, feature = pair.split(':')
        if int(feat_id) in self.feature_filter_dim:
            continue
        
        feat_id = int(feat_id)
        feat_value = float(feature)
        if feat_id not in feat_map:
          feat_map[feat_id] = len(feat_map)
          assert feat_map[feat_id] < self._num_nonzero_feat, '%s features found but %s expected' % (feat_map[feat_id], self._num_nonzero_feat)
        doc_feat[feat_map[feat_id]] = feat_value

      cur_docs.append(doc_feat)
      cur_labels.append(label)

    all_docs = np.concatenate([x['documents'] for x in queries], axis=0)
    all_n_docs = np.array([x['n_docs'] for x in queries], dtype=np.int64)
    all_labels = np.concatenate([x['labels'] for x in queries], axis=0)

    query_ranges = _add_zero_to_vector(np.cumsum(all_n_docs))

    return query_ranges, all_docs, all_labels


  def _create_feature_mapping(self, feature_dict):
    total_features = 0
    feature_map = {}
    for fid in feature_dict:
      if fid not in feature_map:
        feature_map[fid] = total_features
        total_features += 1
    return feature_map

  def _normalize_feat(self, query_ranges, feature_matrix):
    non_zero_feat = np.zeros(feature_matrix.shape[1], dtype=bool)
    for qid in range(query_ranges.shape[0]-1):
      s_i, e_i = query_ranges[qid:qid+2]
      cur_feat = feature_matrix[s_i:e_i,:]
      min_q = np.amin(cur_feat, axis=0)
      max_q = np.amax(cur_feat, axis=0)
      cur_feat -= min_q[None, :]
      denom = max_q - min_q
      denom[denom == 0.] = 1.
      cur_feat /= denom[None, :]
      non_zero_feat += np.greater(max_q, min_q)
    return non_zero_feat

  def read_data(self):
    """
    Reads data from a fold folder (letor format).
    """
    data_read = False
    if self.feature_normalization and self.purge_test_set:
      pickle_name = 'binarized_purged_querynorm.npz'
    elif self.feature_normalization:
      pickle_name = 'binarized_querynorm.npz'
    elif self.purge_test_set:
      pickle_name = 'binarized_purged.npz'
    else:
      pickle_name = 'binarized.npz'

    pickle_path = self.data_path + pickle_name

    train_raw_path = self.data_path + 'train.txt'
    valid_raw_path = self.data_path + 'vali.txt'
    test_raw_path = self.data_path + 'test.txt'

    if self.read_from_pickle and os.path.isfile(pickle_path):
      loaded_data = np.load(pickle_path, allow_pickle=True)
      if loaded_data['format_version'] == FOLDDATA_WRITE_VERSION:
        feature_map = loaded_data['feature_map'].item()
        train_feature_matrix = loaded_data['train_feature_matrix']
        train_doclist_ranges = loaded_data['train_doclist_ranges']
        train_label_vector   = loaded_data['train_label_vector']
        valid_feature_matrix = loaded_data['valid_feature_matrix']
        valid_doclist_ranges = loaded_data['valid_doclist_ranges']
        valid_label_vector   = loaded_data['valid_label_vector']
        test_feature_matrix  = loaded_data['test_feature_matrix']
        test_doclist_ranges  = loaded_data['test_doclist_ranges']
        test_label_vector    = loaded_data['test_label_vector']
        data_read = True
        print("load existing datasets")
      del loaded_data

    if not data_read:
      feature_map = {}
      (train_doclist_ranges,
       train_feature_matrix,
       train_label_vector)  = self._read_file(train_raw_path,
                                              feature_map,
                                              False)
      (valid_doclist_ranges,
       valid_feature_matrix,
       valid_label_vector)  = self._read_file(valid_raw_path,
                                              feature_map,
                                              False)
      (test_doclist_ranges,
       test_feature_matrix,
       test_label_vector)   = self._read_file(test_raw_path,
                                              feature_map,
                                              self.purge_test_set)

      assert len(feature_map) == self._num_nonzero_feat, '%d non-zero features found but %d expected' % (len(feature_map), self._num_nonzero_feat)
      if self.feature_normalization:
        non_zero_feat = self._normalize_feat(train_doclist_ranges,
                                             train_feature_matrix)
        self._normalize_feat(valid_doclist_ranges,
                             valid_feature_matrix)
        self._normalize_feat(test_doclist_ranges,
                             test_feature_matrix)

        list_map = [x[0] for x in sorted(feature_map.items(), key=lambda x: x[1])]
        filtered_list_map = [x for i, x in enumerate(list_map) if non_zero_feat[i]]

        feature_map = {}
        for i, x in enumerate(filtered_list_map):
          feature_map[x] = i

        train_feature_matrix = train_feature_matrix[:, non_zero_feat]
        valid_feature_matrix = valid_feature_matrix[:, non_zero_feat]
        test_feature_matrix  = test_feature_matrix[:, non_zero_feat]

      # sort found features so that feature id ascends
      sorted_map = sorted(feature_map.items())
      transform_ind = np.array([x[1] for x in sorted_map])

      train_feature_matrix = train_feature_matrix[:, transform_ind]
      valid_feature_matrix = valid_feature_matrix[:, transform_ind]
      test_feature_matrix  = test_feature_matrix[:, transform_ind]

      feature_map = {}
      for i, x in enumerate([x[0] for x in sorted_map]):
        feature_map[x] = i

      if self.store_pickle_after_read:
        np.savez_compressed(pickle_path,
                format_version = FOLDDATA_WRITE_VERSION,
                feature_map = feature_map,
                train_feature_matrix = train_feature_matrix,
                train_doclist_ranges = train_doclist_ranges,
                train_label_vector   = train_label_vector,
                valid_feature_matrix = valid_feature_matrix,
                valid_doclist_ranges = valid_doclist_ranges,
                valid_label_vector   = valid_label_vector,
                test_feature_matrix  = test_feature_matrix,
                test_doclist_ranges  = test_doclist_ranges,
                test_label_vector    = test_label_vector,
              )
    if self.shared_resource:
      train_feature_matrix = _make_shared(train_feature_matrix)
      train_doclist_ranges = _make_shared(train_doclist_ranges)
      train_label_vector   = _make_shared(train_label_vector)
      valid_feature_matrix = _make_shared(valid_feature_matrix)
      valid_doclist_ranges = _make_shared(valid_doclist_ranges)
      valid_label_vector   = _make_shared(valid_label_vector)
      test_feature_matrix  = _make_shared(test_feature_matrix)
      test_doclist_ranges  = _make_shared(test_doclist_ranges)
      test_label_vector    = _make_shared(test_label_vector)

    n_feat = len(feature_map)
    assert n_feat == self.num_features, '%d features found but %d expected' % (n_feat, self.num_features)

    self.inverse_feature_map = feature_map
    self.feature_map = [x[0] for x in sorted(feature_map.items(), key=lambda x: x[1])]
    self.train = DataFoldSplit(self,
                               'train',
                               train_doclist_ranges,
                               train_feature_matrix,
                               train_label_vector)
    self.validation = DataFoldSplit(self,
                               'validation',
                               valid_doclist_ranges,
                               valid_feature_matrix,
                               valid_label_vector)
    self.test = DataFoldSplit(self,
                               'test',
                               test_doclist_ranges,
                               test_feature_matrix,
                               test_label_vector)
    self._data_ready = True

def expRelConvert(label,epsilon=0.1):
  """
  This function converts relevance degree from integer number  [0,1,2,...,], to fraction.
  """
  maxLabel=np.max(label)
  label=epsilon+(1-epsilon)*(2**label-1)/(2**maxLabel-1)
  return label
def get_data(dataset,dataset_info_path,fold_id,query_least_size=0,queryMaximumLength=np.inf,\
  relvance_strategy="TrueAverage",voidFeature=True,RelConvertfcn=expRelConvert, rankListLength=5,):
    data = get_dataset_from_json_info(
                  dataset,
                  dataset_info_path,
                  shared_resource = False,
                  relvance_strategy=relvance_strategy
                )
    fold_id = (fold_id-1)%data.num_folds()
    data = data.get_data_folds()[fold_id]
    data.read_data()
    for data_split in [data.test,data.train,data.validation]:
      data_split.filtered_query_sizes(query_least_size,queryMaximumLength)
      data_split.set_relvance_strategy(relvance_strategy)
      data_split.set_rankListLength(rankListLength)
      if voidFeature:
        data_split.voidFeature()
      data_split.label_vector=RelConvertfcn(data_split.label_vector)
    for data_split in [data.train,data.validation]:
      data_split.voidFeature()  ## void them since they are not used.
    return data
def get_query_aver_length(data):
    total_docs=data.train.num_docs()+\
                data.validation.num_docs()
                # data.test.num_docs()+\
                
    total_queries=data.train.num_queries()+\
                  data.validation.num_queries()
                # data.test.num_queries()+\
    return int(total_docs/total_queries)
    # print(total_queries)

def get_data_stat(data,query_least_size=5):
    data.train.filtered_query_sizes(query_least_size)
    data.validation.filtered_query_sizes(query_least_size)
    data.test.filtered_query_sizes(query_least_size)
    total_queries=len(data.train.get_filtered_queries())+\
                len(data.test.get_filtered_queries())+\
                len(data.validation.get_filtered_queries())
    total_docs =np.sum(data.train.query_sizes()[data.train.get_filtered_queries()])+\
                np.sum(data.test.query_sizes()[data.test.get_filtered_queries()])+\
                np.sum(data.validation.query_sizes()[data.validation.get_filtered_queries()])
    feature=data.train.feature_matrix.shape[-1]
    average_num_doc=np.int(np.round(total_docs/total_queries))
    total_irrelevant_docs_orig =np.sum(data.train.label_vector==0)+\
      np.sum(data.test.label_vector==0)+\
        np.sum(data.test.label_vector==0)
    total_docs_orig =data.train.num_docs()+data.validation.num_docs()+data.test.num_docs()
    irrelevant_ratio= total_irrelevant_docs_orig/total_docs_orig   
    return [total_queries,average_num_doc,feature,total_docs,irrelevant_ratio]
def get_mutiple_data_statics(data_name_list=[]):
    stas_list=[]
    for data_name in data_name_list:
        data_setting={"dataset_info_path":"local_dataset_info.txt",
                 "dataset":data_name,
                 "fold_id":1 }
        data=get_data(**data_setting)
        stats=get_data_stat(data)
        stas_list.append(stats)
    df = pd.DataFrame(stas_list,index=data_name_list,columns=["# Queries","# Average documents","# Unique feature","# Total documents","IrrelevantRatio"])
    return df



def clip(value_list,low=0.05,high=0.95):
    result_list=[]
    for value in value_list:
        result_list.append(np.clip(value,low,high))
    return result_list
    