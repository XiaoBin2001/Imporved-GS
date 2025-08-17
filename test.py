import os

paramList = [
    ['bicycle', 3000_000],
]

"""
    ['bicycle', 3000_000],
    ['flowers', 1500_000],
    ['garden', 3000_000],
    ['stump', 3000_000],
    ['treehill', 1500_000],

    ['bonsai', 1000_000],
    ['counter', 1000_000],
    ['kitchen', 1000_000],
    ['room', 1000_000],

    ['drjohnson', 1500_000],
    ['playroom', 1000_000],

    ['train', 1000_000],
    ['truck', 1500_000],
"""

for params in paramList:
    data = params[0]
    output = params[0]
    budget = params[1]

    one_cmd = f'python train.py -s data\\{data} -m output\\{output} --budget {budget}'
    os.system(one_cmd)
    two_cmd = f'python render.py -m output\\{output}'
    os.system(two_cmd)
    three_cmd = f'python metrics.py -m output\\{output}'
    os.system(three_cmd)
    four_cmd = f'python metrics-train.py -m output\\{output}'
    os.system(four_cmd)
