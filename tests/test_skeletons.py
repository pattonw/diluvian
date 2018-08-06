from diluvian.skeletons import *
import random

def test_skeleton():
    test_skeleton = Skeleton()
    try:
        test_skeleton.get_bounds()
    except Exception as e:
        assert str(e) == "Skeleton does not have a positive volume"

    parents = [random.choice(range(max(i,1))) for i in range(10)]
    nodes = [(0,0),(1,0),(2,1),(3,2),(4,0),(5,4),(6,2),(7,2),(8,7),(9,7)]
    test_skeleton.outline_from_pairs(nodes)

    bfs_traversal_order = [0,1,4,2,5,3,6,7,8,9]
    assert bfs_traversal_order == [x.key for x in test_skeleton.tree.traverse()]
    


