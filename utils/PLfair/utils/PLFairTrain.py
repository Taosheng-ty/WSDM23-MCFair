import utils.PLfair.algorithms.PLRank as plr
# import algorithms.pairwise as pw
# import algorithms.lambdaloss as ll
import utils.PLfair.algorithms.tensorflowloss as tfl
# import utils.dataset as dataset
import utils.PLfair.utils.nnmodel as nn
# import utils.evaluate as evl
import utils.PLfair.utils.plackettluce as pl
import argparse
import numpy as np
import time
import tensorflow as tf
import json
def TrainPLFairModel(data_split,metric_weights,rankListLength=5,fairness_tradeoff_param=1):
  model_params = {'hidden units': [32, 32],
                  'learning_rate': 0.1,}
  model = nn.init_model(model_params)
  optimizer = tf.keras.optimizers.SGD(learning_rate=model_params['learning_rate'])
  n_epochs=20
  num_exposure_samples=100
  num_samples=num_exposure_samples
  cutoff=rankListLength
  
  
  for epoch_i in range(n_epochs):
    query_permutation = np.random.permutation(data_split.queriesList)
    for qid in query_permutation:
      q_labels =  data_split.query_values_from_vector(
                                qid, data_split.label_vector)
      q_feat = data_split.query_feat(qid)
      if np.sum(q_labels) > 0 and q_labels.size > 1:
        q_n_docs = q_labels.shape[0]
        q_cutoff = min(cutoff, q_n_docs)
        q_metric_weights = metric_weights[:q_cutoff] #/q_ideal_metric
        with tf.GradientTape() as tape:
          q_tf_scores = model(q_feat)

          q_np_scores = q_tf_scores.numpy()[:,0]

          sampled_rankings = pl.gumbel_sample_rankings(
                                          q_np_scores,
                                          num_exposure_samples,
                                          cutoff=q_cutoff)[0]

          doc_exposure = np.zeros(q_n_docs, dtype=np.float64)
          np.add.at(doc_exposure, sampled_rankings[:,1:], q_metric_weights[1:])
          doc_exposure /= num_exposure_samples

          max_score = np.amax(q_np_scores)
          first_prob = np.exp(q_np_scores-max_score)/np.sum(np.exp(q_np_scores-max_score))
          doc_exposure += first_prob*q_metric_weights[0]

          swap_reward = doc_exposure[:,None]*q_labels[None,:]
          pair_error = (swap_reward - swap_reward.T)
          q_eps = np.mean(pair_error*q_labels[:, None], axis=0)
          q_eps *= 4./(q_n_docs-1.)
          FinalTrainLabel=q_labels*(1-fairness_tradeoff_param)+q_eps*fairness_tradeoff_param
          doc_weights = plr.PL_rank_2(
                                      q_metric_weights,
                                      FinalTrainLabel,
                                      q_np_scores,
                                      sampled_rankings=sampled_rankings[:num_samples,:])
          # else:
          #   raise NotImplementedError('Unknown loss %s' % args.loss)

          loss = -tf.reduce_sum(q_tf_scores[:,0] * doc_weights)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return model