import os
import time
import argparse

import torch
import torch.nn as nn
from conformer import Conformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--sequence_length', default=12345, type=int)
    parser.add_argument('--dim', default=80, type=int)
    args = parser.parse_args()
    print(args)
    return args

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def inference(args):
    batch_size, sequence_length, dim = args.batch_size, args.sequence_length, args.dim
    
    inputs = torch.rand(batch_size, sequence_length, dim).to(args.device)
    input_lengths = torch.LongTensor([12345, 12300, 12000])
    
    model = Conformer(num_classes=10, 
                      input_dim=dim, 
                      encoder_dim=32, 
                      num_encoder_layers=3).to(args.device)
    
    total_time = 0.0
    total_sample = 0

    # Forward propagate
    for i in range(args.num_iter + args.num_warmup):
        elapsed = time.time()
        outputs, output_lengths = model(inputs, input_lengths)
        elapsed = time.time() - elapsed
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        if i >= args.num_warmup:
            total_sample += args.batch_size
            total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def main():
    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    inference(args)

if __name__ == "__main__":
    main()

