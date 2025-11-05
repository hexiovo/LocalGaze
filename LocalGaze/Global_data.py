ALL_MODELS = ['ridge','elastic_net', 'linear_svr',  'svr', 'tiny_mlp']

# 定义每个模型对应的默认参数
MODEL_KWARGS = {
    'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
    'linear_svr': {'C': 1.0, 'epsilon': 0.1},
    'ridge': {'alpha': 1.0},
    'svr': {'C': 1.0, 'epsilon': 0.1},
    'tiny_mlp': {'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'max_iter': 500}
}

smooth_type = "exponential"
# 默认方案，可选 "exponential", "weighted_average", "lowpass", "kalman",推荐exponential和kalman
