import numpy as np


class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx=None):
        """
        TODO: Add summary
        :param left:
        :param right:
        :param is_leaf:
        :param idx:
        """
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx
        if self.left is not None:
            left.parent = self
        if self.right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        """
        TODO: Add summary
        :param value:
        :param idx:
        :return:
        """
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


class SumTree:
    def __init__(self, inp):
        self.root = None
        self.leaf_nodes = None
        self.create_tree(inp)

    def create_tree(self, inp: list):
        """
        TODO: Add Summary
        :param inp:
        :return:
        """
        nodes = [Node.create_leaf(v, i) for i, v in enumerate(inp)]
        leaf_nodes = nodes
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]

        self.root = nodes[0]
        self.leaf_nodes = leaf_nodes

    def retrieve(self, value: float, node: Node):
        """
        TODO: Add summary
        :param value:
        :param node:
        :return:
        """
        if node.is_leaf:
            return node.value, node.idx
        if node.left.value >= value:
            return self.retrieve(value, node.left)
        else:
            return self.retrieve(value - node.left.value, node.right)

    def update(self, idx: int, new_value: float):
        node = self.leaf_nodes[idx]
        change = new_value - node.value
        node.value = new_value
        self.propagate_changes(change, node.parent)

    def propagate_changes(self, change: float, node: Node):
        node.value += change
        if node.parent is not None:
            self.propagate_changes(change, node.parent)


class Memory:
    def __init__(self, size: int, state_size: int):
        """
        TODO: Add summary
        :param size:
        """
        self.size = size
        self.state_size = state_size

        self.states = np.zeros((self.size, self.state_size), dtype=float)
        self.next_states = np.zeros((self.size, self.state_size), dtype=float)
        self.actions = np.zeros(self.size, dtype=int)
        self.rewards = np.zeros(self.size, dtype=float)
        self.terminated = np.zeros(self.size, dtype=bool)
        self.ps = np.zeros(self.size, dtype=float)

        self.idx = 0  # Current empty slot
        self.num_elements = 0
        self.max_p = 1  # Initial condition

        self.tree = SumTree(self.ps)

    def append(self, state, action: int, reward: float, next_state, terminated: bool):
        """
        TODO: Add summary
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param terminated:
        :return:
        """
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.terminated[self.idx] = terminated
        self.ps[self.idx] = self.max_p
        self.tree.update(self.idx, self.max_p)  # Update probability in SumTree

        if self.num_elements < self.size:
            self.num_elements += 1

        self.idx = (self.idx + 1) % self.size  # Update index to next available position

    def sample(self, batch_size: int):
        """
        TODO: Add summary
        :param batch_size:
        :return:
        """
        sample_idxs = np.zeros(batch_size, dtype=int)
        sample_probs = np.zeros(batch_size, dtype=float)
        for i in range(sample_idxs.shape[0]):
            maxval = self.tree.root.value
            rnd = np.random.uniform(0, maxval)
            sample_probs[i], sample_idxs[i] = self.tree.retrieve(rnd, self.tree.root)

        return self.states[sample_idxs], \
               self.actions[sample_idxs], \
               self.rewards[sample_idxs], \
               self.next_states[sample_idxs], \
               self.terminated[sample_idxs], \
               sample_probs, \
               sample_idxs

    def update_probs(self, sample_idxs, probs):
        """
        TODO: Add summary
        :param probs:
        :param sample_idxs:
        :return:
        """
        for i in range(sample_idxs.shape[0]):
            self.tree.update(sample_idxs[i], probs[i])

        # Update max p
        max_val = np.max(probs)
        if max_val > self.max_p:
            max_p = max_val


def demonstrate_sampling(tree: SumTree):
    root_node = tree.root
    tree_total = root_node.value
    iterations = 1000000
    selected_vals = []
    for i in range(iterations):
        rand_val = np.random.uniform(0, tree_total)
        selected_val = tree.retrieve(rand_val, root_node)
        selected_vals.append(selected_val)

    return selected_vals


if __name__ == '__main__':
    # inp = np.array([1, 4, 2, 3])
    # tree = SumTree(inp)
    # selected_vals = demonstrate_sampling(tree)
    # print(f"Should be ~4: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")
    #
    # tree.update(1, 6)
    # selected_vals = demonstrate_sampling(tree)
    # print(f"Should be ~6: {sum([1 for x in selected_vals if x == 6]) / sum([1 for y in selected_vals if y == 1])}")
    # # the below print statement should output ~2
    # print(f"Should be ~2: {sum([1 for x in selected_vals if x == 6]) / sum([1 for y in selected_vals if y == 3])}")

    n = 10
    mem = Memory(size=n, state_size=2)

    for i in range(n):
        state = np.random.rand(2)
        action = np.random.randint(0, 4)
        reward = np.random.rand()
        next_state = np.random.rand(2)
        terminated = False
        mem.append(state, action, reward, next_state, terminated)

    print(mem.sample(3))
