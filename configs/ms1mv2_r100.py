from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = (0.1*config.batch_size*4)/(1024)
config.verbose = 2000
config.dali = False

config.rec = "/train_tmp/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 28
config.warmup_epoch = 0
config.val_targets = []
config.eta_scale = 0.1
config.eta_t = 0.1
config.eta_theta = 0.1
config.ratio = 0.75
