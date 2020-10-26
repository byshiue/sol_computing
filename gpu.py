class Gpu:
    def __init__(self, computing_ability, memory_bandwidth):
        
        self.com_ab = computing_ability # FLOPs
        self.mem_band = memory_bandwidth # Bytes/s

gpu_v100 = Gpu(125*1e12, 900*1024*1024*1024)
gpu_t4 = Gpu(65*1e12, 320*1024*1024*1024)
gpu_a100 = Gpu(312.5*1e12, 1600*1024*1024*1024)