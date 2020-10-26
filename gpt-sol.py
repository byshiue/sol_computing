class Gpu:
    def __init__(self, computing_ability, memory_bandwidth):
        
        self.com_ab = computing_ability # FLOPs
        self.mem_band = memory_bandwidth # Bytes/s

class Layer:
    def __init__(self, name, comp_cost, mem_cost):
        self.name = name
        self.comp_cost = comp_cost
        self.mem_cost = mem_cost
            
    def mem_time(self, gpu):
        time_in_sec = self.mem_cost * 1.0 / gpu.mem_band
        time_in_usec = time_in_sec * 1e6
        return time_in_usec
    
    def comp_time(self, gpu):
        time_in_sec = self.comp_cost * 1.0 / gpu.com_ab
        time_in_usec = time_in_sec * 1e6
        return time_in_usec
    
    def max_time(self, gpu):
        return max(self.mem_time(gpu), self.comp_time(gpu))
    
class Gpt2_Model:
    def __init__(self, name, gpu, batch, seq, head, size, dtype, num_layer, vocab):
        self.name = name
        self.gpu = gpu
        self.batch = batch
        self.seq = seq
        self.head = head
        self.size = size
        self.hidden = self.head * self.size 
        self.dtype = dtype
        self.dtype_byte= self.dtype / 8
        self.num_layer = num_layer
        self.vocab = vocab
        
        self.build_model()
        self.print_sol()
    
    def build_model(self):
        self.layers = [] 
        self.layers.append(Layer("embedding_lookup",
                                 comp_cost = 0,
                                 mem_cost = self.dtype_byte * (self.vocab * self.hidden + self.batch * self.hidden) ))
        self.layers.append(Layer("transformer/query", 
                                comp_cost = self.num_layer * self.batch * self.hidden * self.hidden,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * self.hidden + self.hidden * self.hidden + self.batch * self.hidden)))
        self.layers.append(Layer("transformer/key", 
                                comp_cost = self.num_layer * self.batch * self.hidden * self.hidden,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * self.hidden + self.hidden * self.hidden + self.batch * self.hidden)))
        self.layers.append(Layer("transformer/value", 
                                comp_cost = self.num_layer * self.batch * self.hidden * self.hidden,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * self.hidden + self.hidden * self.hidden + self.batch * self.hidden)))
        self.layers.append(Layer("transformer/multihead_attention",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/atten_out", 
                                comp_cost = self.num_layer * self.batch * self.hidden * self.hidden,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * self.hidden + self.hidden * self.hidden + self.batch * self.hidden)))
        self.layers.append(Layer("transformer/reduce_communication",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/layer_nom_with_residual",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/ffn gemm 1", 
                                comp_cost = self.num_layer * self.batch * self.hidden * self.hidden * 4,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * self.hidden + self.hidden * 4 * self.hidden + self.batch * 4 * self.hidden)))
        self.layers.append(Layer("transformer/add_bias_gelu",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/ffn gemm 2", 
                                comp_cost = self.num_layer * self.batch * self.hidden * 4 * self.hidden,
                                mem_cost = self.num_layer * self.dtype_byte * (self.batch * 4 * self.hidden + 4 * self.hidden * self.hidden + self.batch * self.hidden)))
        self.layers.append(Layer("transformer/reduce_communication",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/add_bias_input",
                                 comp_cost = 0,
                                 mem_cost = 0))
        
        self.layers.append(Layer("layer_norm",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("compute_logits",
                                 comp_cost = self.batch * self.hidden * self.vocab,
                                 mem_cost = self.dtype_byte * (self.batch * self.hidden + self.hidden * self.vocab + self.batch * self.vocab) ))
        self.layers.append(Layer("top k",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("softmax",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("top p",
                                 comp_cost = 0,
                                 mem_cost = 0))
        
    def print_sol(self):
        total_time = 0
        for l in self.layers:
            total_time += l.max_time(self.gpu)
            
        for l in self.layers:
            print("{:45s} time cost {:6.6f} us ({:2.2f}%)".format(l.name, l.max_time(self.gpu), l.max_time(self.gpu) / total_time * 100))
            
        print("{} model total time cost: {} ms".format(self.name, total_time / 1000))
    

if __name__ == "__main__":
    gpu_v100 = Gpu(125*1e12, 900*1024*1024*1024)
    gpu_t4 = Gpu(65*1e12, 320*1024*1024*1024)
    gpu_a100 = Gpu(312.5*1e12, 1600*1024*1024*1024)
    
    
    gpt2_model = Gpt2_Model(name="GPT-3-v1", gpu=gpu_v100, batch=8, seq=1024, head=56, size=128, dtype=16, num_layer=32, vocab=50257)
    # gpt_2 = Model(gpu_v100)
    # gpt_2.append_layer(GemmLayer(batch*seq, hidden_dim, hidden_dim, dtype))
    # print(gpt_2.total_cost())
    