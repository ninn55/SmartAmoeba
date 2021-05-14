#!/usr/bin/python
# coding=utf-8



class MemPlaner(object):
    def __init__(self):
        super(MemPlaner, self).__init__()
        self.memsize = 0
        self.mempool = [[0, 0, 0, 0],] #[state, index, size, address]
        #state 0: free, 1, alloc

        self.mem_alloc_method = 0
        self.alloc_index = 0
        self.debug = 1

        self.mem_overhead = 16
        self.mem_minsize = 32

    def __get_first_fit(self, memsize):
        index = 0
        for mem in self.mempool:
            if mem[0] == 0:
                if mem[2] >= memsize:
                    return index
            index += 1
        return -1



    def alloc(self, memsize, align = 4):
        memsize = int((memsize + align - 1.0) / align) * align + self.mem_overhead
        if memsize < self.mem_minsize:
            memsize = self.mem_minsize
        self.alloc_index += 1
        index = self.__get_first_fit(memsize)
        alloc_mem = None
        if index == -1:
            alloc_mem = self.mempool[-1]
            if alloc_mem[0] == 1:
                alloc_mem = [1, self.alloc_index, memsize, 0]
                self.mempool.append(alloc_mem)
            else:
                alloc_mem[0] = 1
                alloc_mem[1] = self.alloc_index 
                alloc_mem[2] = memsize
        else:
            memblock = self.mempool[index]
            if memblock[2] >= memsize:
                free = memblock[2] - memsize
                if free == 0:
                    memblock[0] = 1
                    memblock[1] = self.alloc_index
                    alloc_mem = memblock
                else:
                    alloc_mem = [1, self.alloc_index, memsize, 0]
                    memblock[0] = 0
                    memblock[1] = 0
                    memblock[2] = free
                    self.mempool.insert(index, alloc_mem)
        
        if self.debug == 1:
            print('alloc start>')
            for mem in self.mempool:
                print(mem)
            print('<end', alloc_mem)


        return alloc_mem.copy()

    def free(self, ptr):
        for mem in self.mempool:
            if mem[1] == ptr[1]:
                mem[1] = 0
                mem[0] = 0
        
        #Merge adjacent memory region
        index = 0
        try:
            while True:
                mem = self.mempool[index]
                mem_next = self.mempool[index + 1]
                if (mem[0] == 0) and (mem_next[0] == 0):
                    mem[2] += mem_next[2]
                    self.mempool = self.mempool[0:index+ 1] + self.mempool[index+2:]
                else:
                    index += 1

        except:
            pass

        if self.debug == 1:
            print('free start>')
            for mem in self.mempool:
                print(mem)
            print('<end')

    def getsize(self):
        sum_size = 0

        for mem in self.mempool:
            sum_size += mem[2]

        return (sum_size + self.mem_overhead)








if __name__ == '__main__':
    mp = MemPlaner()
    p1 = mp.alloc(1000)
    p2 = mp.alloc(1000)
    p3 = mp.alloc(1000)
    p4 = mp.alloc(2200)
    mp.free(p1)
    p5 = mp.alloc(1500)
    mp.free(p4)
    p6 = mp.alloc(3000)
    mp.free(p5)
    mp.free(p2)
    mp.free(p3)


    print(mp.getsize())