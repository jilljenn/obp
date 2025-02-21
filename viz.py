import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def viz_grid_policy(grid, title):
    displayed = pd.DataFrame(grid[:, :, 0], columns=range(1, n_actions + 1), index=all_contexts.flatten())
    # plt.imshow(action_grid[:, sorted_order, 0])
    sns.heatmap(displayed, cmap='cool')
    plt.xlabel('items ranked by difficulties →')
    plt.ylabel('← context')
    plt.title(title)

viz_grid_policy(
    gbdt_policy[df.reset_index().groupby('context').first()['index'].values],
    'Optimal GBDT'
)
# plt.savefig('obp/gbdt_policy.png')

all_contexts = np.array(sorted(df['context'].unique())).reshape(-1, 1)

def viz_policy_grid(grid, title, filename, cbar=False, cmap='magma', figsize=(5, 5)):
    plt.tight_layout()
    plt.style.use('default')
    plt.subplots(figsize=figsize)
    sns.heatmap(grid, cmap=cmap, cbar=cbar, xticklabels=False, yticklabels=False)
    plt.xlabel('actions ranked by difficulties →')
    plt.ylabel('← context')
    plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def viz_policy(policy, title, filename=None):
    action_grid = policy.predict(context=all_contexts)
    displayed = pd.DataFrame(action_grid[:, :, 0], columns=range(1, n_actions + 1), index=all_contexts.flatten())
    viz_policy_grid(displayed, title, filename)
    # plt.imshow(action_grid[:, sorted_order, 0])
    # plt.style.    
    
viz_policy(eval_policy, title='Optimal policy found for $\widehat{V}_{\mathsf{IPW}}$ using Random Forests',
           filename=f'{DATA}-rf-ipw-context-elo.png')

def query_policy_in_test(policy_realizations):
    df_test['chosen_action'] = policy_realizations.argmax(axis=1).reshape(-1)
    subset = df_test.query("action == chosen_action")
    print(f"Found {len(subset)} occurrences ({len(subset) / len(df_test):.1f}%), mean reward = {subset['reward'].mean()}")

# all_contexts.shape
def display_policy(policy, title='My policy', filename=None):
    policy_grid = np.zeros((len(all_contexts), n_actions))
    policy_grid[range(len(all_contexts)), policy.values] = 1

    viz_policy_grid(policy_grid, title, filename)
    """displayed = pd.DataFrame(my_ipw_grid, columns=range(1, n_actions + 1), index=all_contexts.flatten())
    sns.heatmap(displayed, cmap='cool')
    plt.xlabel('items ranked by difficulties →')
    plt.ylabel('← context')
    plt.title(title)"""
    
"""my_ipw_grid = np.zeros((len(all_contexts), n_actions))
my_ipw_grid[range(len(all_contexts)), my_ipw.values] = 1
displayed = pd.DataFrame(my_ipw_grid, columns=range(1, n_actions + 1), index=all_contexts.flatten())
sns.heatmap(displayed, cmap='cool')
plt.xlabel('items ranked by difficulties →')
plt.ylabel('← context')
plt.title('Unbounded my IPW policy')"""
