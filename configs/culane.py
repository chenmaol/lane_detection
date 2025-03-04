# DATA
dataset='CULane'
data_root = ['wrcg_data/belgium','wrcg_data/germany', 'wrcg_data/monte_carlo', 'wrcg_data/spain', 'wrcg_data/sweden', 'wrcg_data/argentina', 'wrcg_data/chile', 'wrcg_data/corsica', 'wrcg_data/croatia', 'wrcg_data/estonia', 'wrcg_data/finland', 'wrcg_data/greece', 'wrcg_data/japan', 'wrcg_data/kenya', 'wrcg_data/mexico', 'wrcg_data/new_zealand', 'wrcg_data/portugal', 'wrcg_data/sanremo', 'wrcg_data/sardinia', 'wrcg_data/turkey', 'wrcg_data/wales' ]

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'Adam'  #['SGD','Adam']
learning_rate = 0.0001
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = './result'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = 'result/unified_0227/ep049.pth'
test_work_dir = None

num_lanes = 4




