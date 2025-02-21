from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from obp.policy import IPWLearner
from obp.policy import QLearner
from sklearn.ensemble import GradientBoostingRegressor
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedInverseProbabilityWeighting as SNIPW,
    SwitchDoublyRobust as SDR,
)
import sklearn
from collections import Counter
from prepare import sigmoid
import pickle
import numpy as np
import logging
import pandas as pd


MINIMAX = ['max_context', 'min_context', 'prev_outcome']


# from sklearn.utils.estimator_checks import check_estimator
# DKT
# theta_0 : DKT

class AlmostIRTreward(sklearn.base.BaseEstimator):
    def __init__(self, difficulties):
        super().__init__()
        self.difficulties = difficulties
        self.logreg = LogisticRegression(solver='liblinear')  # , C=1e10

    def fit(self, X, y):
        # print(Counter(X[:10, 1]), Counter(y[:10]))
        # print(Counter(y))
        self.logreg.fit(X, y > 0)
    
    def predict(self, X):
        # print(X[:, 0])
        # print(X[:, 2])
        # print(X[:, -1])
        return self.logreg.predict_proba(X)[:, 1] * (X[:, -1] - self.difficulties.min())


class IRTreward(sklearn.base.BaseEstimator):
    def __init__(self, difficulties):
        super().__init__()
        self.difficulties = difficulties

    def fit(self, X, y):
        # print(X[0], df.loc[0])
        pass
    
    def predict(self, X):
        return sigmoid(X[:, 0] - X[:, 1]) * (X[:, 1] - self.difficulties.min())


def pd_to_dict(df, keys, CONTEXT):
    d = {key: df[key].to_numpy() for key in keys}
    if CONTEXT == 'minimax':
        d['context'] = df[MINIMAX]
    else:
        d['context'] = df[['context']]
    return d


def pad_policy(df_policy, n_actions):
    for action in range(n_actions):
        if action not in df_policy.columns:
            df_policy[action] = 0
    return df_policy.reindex(sorted(df_policy.columns), axis=1)


def compute_optimal_ipw(df, n_actions, CONTEXT):
    if CONTEXT == 'minimax':
        my_ipw = df.groupby(MINIMAX + ['context_id', 'action'])['reward_on_pscore'].sum().reset_index().sort_values(
            'reward_on_pscore', ascending=False).groupby(['context_id'])['action'].first()

        ipw_mapping = dict(zip(my_ipw.index, my_ipw.values))

        df['ipw_best_action'] = df['context_id'].map(ipw_mapping)

        indices = df['ipw_best_action']
        logging.warning(indices)

    else:
        df['reward_on_pscore'] = df['reward'] / df['pscore']

        my_ipw = df.groupby(['context', 'action'])['reward_on_pscore'].sum().reset_index().sort_values(
            'reward_on_pscore', ascending=False).groupby('context')['action'].first()
        indices = my_ipw.loc[df['context']].values  # _train
        logging.warning(indices)

    my_ipw_actions = np.zeros((len(indices), n_actions))
    my_ipw_actions[range(len(indices)), indices] = 1
    my_ipw_actions = my_ipw_actions.reshape(-1, n_actions, 1)
    return my_ipw_actions


def get_context(df, CONTEXT):
    if CONTEXT == 'minimax':
        return df[MINIMAX].to_numpy()
    else:
        return df["elo_context"].to_numpy().reshape(-1, 1)


def compute_gbdt(df, n_actions, difficulties, CONTEXT):
    context = get_context(df, CONTEXT)
    print(context.shape)
    qlearner=QLearner(
        n_actions=n_actions,
        # base_model=RandomForestRegressor(n_estimators=50)
        base_model=GradientBoostingRegressor(n_estimators=50, max_depth=5),
        # base_model=AlmostIRTreward()
    )
    qlearner.q_estimator.action_context=difficulties.reshape(-1, 1)
    qlearner.fit(
        # context=df[["dim1", "dim2"]].to_numpy(),
        context=context,
        action=df["action"].to_numpy(),
        reward=df["reward"].to_numpy(),
        pscore=df["pscore"].to_numpy()
        # pscore=new_pscores
    )
    qlearner_dist=qlearner.predict(
        # context=df["elo_context"].to_numpy().reshape(-1, 1)
        context=context,
        # context=df[["dim1", "dim2"]].to_numpy()
    )
    # Robo: 13 s perf
    # Assist: 11 s / GBDT 9 s (qt bin) / GBDT 8 s / RF 33 s / GBDT 50 5: 13 s
    # Assist Elo: 20 s
    # Assist Minimax: 18 s
    # Dim2: 1 s
    # RoboMinimax: 15 s
    return qlearner_dist

def compute_irt(df, n_actions, difficulties, CONTEXT):
    context = get_context(df, CONTEXT)
    irt_qlearner=QLearner(
        n_actions=n_actions,
        base_model=IRTreward(difficulties),
    )
    irt_qlearner.q_estimator.action_context=difficulties.reshape(-1, 1)
    irt_qlearner.fit(
        context=context,
        action=df["action"].to_numpy(),
        reward=df["reward"].to_numpy(),
        pscore=df["pscore"].to_numpy()
    )
    irt_qlearner_dist=irt_qlearner.predict(
        context=context)
    return irt_qlearner_dist

# def compute_gbdt%%time

#     irt_regression_model = RegressionModel(
#         n_actions=n_actions,
#         base_model=IRTreward(),
#         action_context=difficulties.reshape(-1, 1)
#     )
#     irt_regression_model.fit(
#         context=df["elo_context"].to_numpy().reshape(-1, 1),
#         action=df["item"].to_numpy(),
#         reward=df["reward"].to_numpy(),
#     )
#     irt_estimated_rewards = irt_regression_model.predict(
#         context=df["elo_context"].to_numpy().reshape(-1, 1),
#     )
#     # RoboMission: 34 s perf / IRT 9 s
#     # Assist: 56 s perf / Almost IRT 24 s

def compute_gbdt_estimator(df, n_actions, difficulties, CONTEXT):
    context = get_context(df, CONTEXT)
    # In dim 2, best is MultinomialReg
    regression_model = RegressionModel(
        n_actions=n_actions,
        # base_model=RandomForestRegressor(n_estimators=50),
        base_model=GradientBoostingRegressor(n_estimators=50, max_depth=5),
        # base_model=AlmostIRTreward(),
        # base_model=IRTreward(),
        action_context=difficulties.reshape(-1, 1)
    )

    regression_model.fit(
        # context=dkt_df[["dim1", "dim2"]].to_numpy().reshape(-1, 2),
        # context=df[["elo_context"]].to_numpy(),
        context=context,
        action=df["action"].to_numpy(),
        reward=df["reward"].to_numpy(),
    )
    estimated_rewards_by_reg_model = regression_model.predict(
        # context=df["elo_context"].to_numpy().reshape(-1, 1),
        context=context,
        # context=df[["dim1", "dim2"]].to_numpy().reshape(-1, 2)
        # action=df_test["item"].to_numpy(),
        # reward=df_test["reward"].to_numpy(),
    )
    # RoboMission: 12 s perf
    # Assist: 56 s perf / Almost IRT 24 s / mais avec les difficultés GBDT 9 s / optimal bins 8 s / RF 50 33 s
    # Assist Elo: 20 s
    # Assist Minimax: 18 s
    # GBDT 50 5: 12 s
    # Dim2 : 1 s
    # RoboMinimax: 15 s
    return estimated_rewards_by_reg_model


def compute_behavior(df):
    # blabla = pd.DataFrame()
    # for _, row in df[['context_id', 'item', 'pscore']].drop_duplicates().iterrows():
    #     blabla.loc[row['context_id'], row['item']] = row['pscore']
    # # Assist Minimax: 53 s

    # behavior_policy = blabla.reindex(sorted(blabla.columns), axis=1)#.sum(axis=1)
    # behavior_policy.shape

    # repeated_behavior_policy = behavior_policy.loc[df['context_id']]
    # repeated_behavior_policy.shape

    behavior_policy = df[['context_id', 'action', 'pscore']].drop_duplicates().pivot(
        index='context_id', columns='action', values='pscore').fillna(0.)
    repeated_behavior_policy = behavior_policy.loc[df['context_id']]
    return repeated_behavior_policy


def compute_closest50(df, n_actions, difficulties, CONTEXT):
    # Compute closest item for minimax
    mid_contexts = df['mid_context'].drop_duplicates().values
    if CONTEXT == 'minimax':
        closest = abs(mid_contexts[:, None] - (difficulties - difficulties.min())).argmin(axis=1)
    else:
        closest = abs(mid_contexts[:, None] - difficulties).argmin(axis=1)
    closest_mapping = dict(zip(mid_contexts, closest))

    df['closest_middle'] = df['mid_context'].map(closest_mapping)
    closest_middle_actions = pd.get_dummies(df['closest_middle'], columns=range(n_actions), dtype=int)
    closest_middle_actions = pad_policy(closest_middle_actions, n_actions).values.reshape(-1, n_actions, 1)
    return closest_middle_actions


def compute_random(df, n_actions):
    return np.ones((len(df), n_actions, 1)) * 1 / n_actions


def save(policies, filename):
    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(policies, f)


def compute_policies(df, n_actions, difficulties, CONTEXT):
    logging.warning('Compute IPW')
    my_ipw_actions = compute_optimal_ipw(df, n_actions, CONTEXT)
    logging.warning('Compute GBDT')
    qlearner_dist = compute_gbdt(df, n_actions, difficulties, CONTEXT)
    irt_qlearner_dist = None
    if CONTEXT == 'elo':
        logging.warning('Compute IRT')
        irt_qlearner_dist = compute_irt(df, n_actions, difficulties, CONTEXT)
    logging.warning('Compute Behavior')
    repeated_behavior_policy = compute_behavior(df)
    logging.warning('Compute Random')
    random_policy = compute_random(df, n_actions)
    logging.warning('Compute Closest to 50%')
    closest_middle_actions = compute_closest50(df, n_actions, difficulties, CONTEXT)
    return my_ipw_actions, qlearner_dist, irt_qlearner_dist, repeated_behavior_policy, random_policy, closest_middle_actions


def backup_data(df, n_actions, my_ipw_actions, qlearner_dist, irt_qlearner_dist, repeated_behavior_policy, random_policy, closest_middle_actions, estimated_rewards_by_reg_model, DATA, CONTEXT):

    policies = [
        ('Optimal IPW', my_ipw_actions),
        ('GBDT', qlearner_dist),
        
        # ('Optimal IRT', unbounded_policy_test_grid.reshape(-1, n_actions, 1)),
        ('Behavior', repeated_behavior_policy.fillna(0.).values.reshape(-1, n_actions, 1)),
        # ('Behavior', full_behavior_policy.reshape(-1, n_actions, 1))
        ('Closestmiddle', closest_middle_actions),
        ('Random', random_policy)
    ]
    if CONTEXT == 'elo':
        policies.append(('Optimal IRT', irt_qlearner_dist))

    backup = {
        'DATA': DATA,
        'feedback': pd_to_dict(df, ['action', 'pscore', 'reward', 'position'], CONTEXT),
        'policies': policies,
        'estimated_rewards_by_reg_model': estimated_rewards_by_reg_model
    }

    save(backup, f'{DATA}-{CONTEXT}-new')
