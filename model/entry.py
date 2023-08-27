import multiprocessing
import os
import load_data_distributed
# import configs
import torch.distributed as dist
def main(rank, world_size):
    # Set environment variables and initialize DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # dist.init_process_group(backend='nccl')
    load_data_distributed.run()
    # Rest of your training code goes here

if __name__ == "__main__":
    world_size = 4  # Assuming you have 4 GPUs
    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(target=main, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
