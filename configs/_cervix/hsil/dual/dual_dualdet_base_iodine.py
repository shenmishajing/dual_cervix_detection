_base_ = [
    './dual_dualdet_base.py'
]
data = dict(
    train=dict(prim='iodine'),
    val=dict(prim='iodine'),
    test=dict(prim='iodine'))