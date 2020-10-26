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

