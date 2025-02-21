from prepare import get_data, populate_data, compute_contexts, compute_pscore
from learn import compute_policies, compute_gbdt_estimator, backup_data
import logging
import sys


DATA = sys.argv[1]
CONTEXT = sys.argv[2]
logging.warning('Running %s %s', DATA, CONTEXT)


df = get_data(DATA)
n_actions, difficulties = populate_data(df, DATA)
compute_contexts(df, difficulties, CONTEXT)
compute_pscore(df, CONTEXT)
my_ipw_actions, qlearner_dist, irt_qlearner_dist, repeated_behavior_policy, random_policy, closest_middle_actions = compute_policies(df, n_actions, difficulties, CONTEXT)
estimated_rewards_by_reg_model = compute_gbdt_estimator(df, n_actions, difficulties, CONTEXT)
backup_data(df, n_actions, my_ipw_actions, qlearner_dist, irt_qlearner_dist, repeated_behavior_policy, random_policy, closest_middle_actions, estimated_rewards_by_reg_model, DATA, CONTEXT)
