from utils.gpu import gpu_v100
from utils.gpu import gpu_a100
from utils.gpu import gpu_t4
from utils.layer import Layer

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
        
        m = self.batch
        k = self.hidden
        n = self.hidden
        self.layers.append(Layer("transformer/query", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        self.layers.append(Layer("transformer/key", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        self.layers.append(Layer("transformer/value", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        
        # multihead attention uses "(1 + self.seq) / 2 because the size grows from 1 to self.seq"
        self.layers.append(Layer("transformer/multihead_attention",
                                 comp_cost = self.batch * self.head * (self.size * 2 * (1 + self.seq) / 2), # only consider batch gemm ignore softmax cost
                                 mem_cost = self.num_layer * self.dtype_byte * (2 * m * n + 2 * m * (1 + self.seq) / 2 * n)))
        self.layers.append(Layer("transformer/atten_out", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        self.layers.append(Layer("transformer/reduce_communication",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/layer_nom_with_residual",
                                 comp_cost = 0,
                                 mem_cost = 0))
        n = 4 * self.hidden
        self.layers.append(Layer("transformer/ffn gemm 1", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        self.layers.append(Layer("transformer/add_bias_gelu",
                                 comp_cost = self.num_layer * 16 * m * n, # Assume cost of tanh is 10 times of float point 
                                 mem_cost = self.num_layer * self.dtype_byte * (2 * m * n + n)))
        k = 4 * self.hidden
        n = self.hidden
        self.layers.append(Layer("transformer/ffn gemm 2", 
                                comp_cost = self.num_layer * m * k * n,
                                mem_cost = self.num_layer * self.dtype_byte * (m * k + k * n + m * n)))
        k = self.hidden
        self.layers.append(Layer("transformer/reduce_communication",
                                 comp_cost = 0,
                                 mem_cost = 0))
        self.layers.append(Layer("transformer/add_bias_input",
                                 comp_cost = self.num_layer * 2 * m * n,
                                 mem_cost = self.num_layer * self.dtype_byte * (3 * m * n + n)))
        
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
        total_time_per_step = 0
        for l in self.layers:
            total_time_per_step += l.max_time(self.gpu)
        
        print("{} model analysis: ".format(self.name))
        print("| Layer name {:34s} | time cost (us) | time cost percentage (%) |".format(" "))
        for l in self.layers:
            print("| {:45s} | {:14.6f} | {:24.2f} |".format(l.name, l.max_time(self.gpu), l.max_time(self.gpu) / total_time_per_step * 100))
            
        parameters_num = (12 * self.hidden * self.hidden + 15 * self.hidden) * self.num_layer
        print("{} model".format(self.name))
        print("    Model size: {:5.2f} billion parameters, {} GBs. ".format(parameters_num / 1e9, parameters_num * self.dtype_byte / 1024 / 1024 /1024))
        
        print("    Model time: time cost per step: {:10.6} ms; total time cost: {:10.3} ms \n".format(total_time_per_step / 1000, total_time_per_step * self.seq / 1000))

if __name__ == "__main__":
    # gpt2_model = Gpt2_Model(name="GPT-3-v1", gpu=gpu_v100, batch=8, seq=1024, head=56, size=128, dtype=16, num_layer=32, vocab=50257)
    # gpt2_model = Gpt2_Model(name="GPT-3-v1", gpu=gpu_v100, batch=8, seq=1024, head=96, size=128, dtype=16, num_layer=96, vocab=50257)
    gpt2_model = Gpt2_Model(name="GPT-3 60B-1", gpu=gpu_v100, batch=8, seq=1024, head=96, size=128, dtype=16, num_layer=32, vocab=50257)
    gpt2_model = Gpt2_Model(name="GPT-3 60B-2", gpu=gpu_v100, batch=8, seq=1024, head=56, size=128, dtype=16, num_layer=96, vocab=50257)
    