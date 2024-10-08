
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0
config.fp16 = True
config.weight_decay = 5e-4
config.batch_size = 128
config.optimizer = "sgd"
config.ngpus = 4
config.lr = (0.1*config.batch_size*config.ngpus)/(1024)
config.verbose = 2000
config.dali = False

config.rec = "/train_tmp/WebFace12M"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = []
config.eta_scale = 0.1
config.eta_t = 0.1
config.eta_theta = 0.1
config.ratio = 0.75
