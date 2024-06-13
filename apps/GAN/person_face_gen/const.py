#数据路径
dateset_dir = "/Users/bytedance/Project/personalProject/deepL/apps/dataSet"
#每次训练的batch大小
batch_size = 128
# 图片尺寸
image_size = 64
# 数据load多线程处理数
workers = 2
# 生成器特征数
ngf = 64
# 判别器特征数
ndf = 64
# z潜在向量的大小
nz = 100
# 训练的通道数
nc=3

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002 

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5