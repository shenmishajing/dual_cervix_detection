_base_ = [
    './dual_base.py'
]
data = dict(
    train=dict(prim='acid'),
    val=dict(prim='acid'),
    test=dict(prim='acid'))