# =============================
# From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# =============================
class BaseDataLoader():
    def __init__(self):
        self.opt = None

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None