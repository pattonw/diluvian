
from collections import deque


class Arbor:
    """
    A basic arbor structure. Only has access to nodes and edges
    and provides as much functionality it can with just that.
    Any other data can be stored in the value parameter of
    individual nodes.
    """

    def __init__(self, root=None):
        """
        Initialize an empty tree
        """
        self.root = root

    def search(self, key):
        for node in self.traverse():
            if key == node.get_key():
                return node
        raise Exception("node {0} does not exist".format(key))

    def get_key_map(self):
        key_map = {}
        for node in self.traverse():
            key_map[node.key] = node
        return key_map

    def traverse(self, fifo=False):
        """
        Iterate over the elements of the tree
        traversal options:
            - first in first out (depth first)
            - first in last out (bredth first)
            """
        if self.root is None:
            raise Exception("this arbor has no root")
        else:
            if fifo:
                return self.depth_first_traversal()
            else:
                return self.breadth_first_traversal()

    def traverse_segments(self):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.popleft()
            for child in current.get_children():
                segment = [current, child]
                next_node = segment[-1].get_following(segment[-2])
                while len(next_node) > 0:
                    if len(next_node) > 1:
                        queue.append(segment[-1])
                        break
                    else:
                        segment.append(next_node[0])
                        next_node = segment[-1].get_following(segment[-2])

                yield (segment)

    def get_minimal_subtree(self, ids):
        """
        get the smallest possible subtree containing all given ids
        """
        uncovered = ids
        all_nodes = []
        potential_queue = []
        last = None
        for node in self.traverse(True):
            while node.parent != last and len(potential_queue) > 0:
                del potential_queue[0]
                if len(potential_queue) > 0:
                    last = potential_queue[0]
            potential_queue.insert(0, node)

            if node.key in uncovered:
                uncovered.remove(node.key)
                all_nodes = all_nodes + potential_queue
                potential_queue = []
                last = node

            if last is not None:
                last = node

        assert len(uncovered) == 0, "Not all nodes were found. missing: {}".format(
            uncovered
        )
        return all_nodes

    def get_root_leaf_paths(self):
        potential_queue = []
        last = None
        for node in self.traverse(True):
            while node.parent != last and len(potential_queue) > 0:
                del potential_queue[0]
                if len(potential_queue) > 0:
                    last = potential_queue[0]
            potential_queue.insert(0, node)
            last = node
            if len(node.children) == 0:
                yield potential_queue

            if last is not None:
                last = node

    def breadth_first_traversal(self):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.popleft()
            yield current
            for child in current.get_children():
                queue.append(child)

    def depth_first_traversal(self):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.popleft()
            yield current
            for child in current.get_children():
                queue.insert(0, child)

    class Node:
        """
        Basic Node datastructure, has basic getter and setter methods
        """

        def __init__(self, nid, value=None):
            """
            node has only key, value, parent
            """
            self.key = nid
            self.value = value
            self.parent = None
            self.children = []

        def get_key(self):
            return self.key

        def get_value(self):
            return self.value

        def set_value(self, value):
            self.value = value

        def get_parent(self):
            return self.parent

        def set_parent(self, parent):
            self.parent = parent

        def get_children(self):
            return self.children

        def set_children(self, children):
            self.children = children

        def add_child(self, child):
            self.children.append(child)
            child.set_parent(self)

        def get_neighbors(self):
            if self.parent is not None:
                return [self.parent] + self.children
            else:
                return self.children

        def clone(self):
            """
            cloning functionality for creating
            new arbors based on this one.
            """
            clone = type(self)(self.key, self.value)
            ids = [
                self.key,
                self.parent.key if self.parent is not None else None,
                [child.key for child in self.children],
            ]
            return clone, ids

        def get_following(self, previous):
            """
            get the next node from the perspective of the previous.
            i.e. given nodes:
                a--b--c
            b.get_following(a) = c
            b.get_vollowing(c) = a
            """
            neighbors = self.get_neighbors()
            if len(neighbors) == 2 and previous in neighbors:
                return [neighbors[1 - neighbors.index(previous)]]
            elif len(neighbors) == 1:
                return []
            elif len(neighbors) > 2:
                neighbors.remove(previous)
                return neighbors
            else:
                raise Exception("This node has {} neighbors".format(len(neighbors)))

        def traverse(self):
            queue = deque([self])
            while len(queue) > 0:
                current = queue.popleft()
                yield current
                for child in current.get_children():
                    queue.insert(0, child)

