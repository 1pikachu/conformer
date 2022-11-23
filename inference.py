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

    model.eval()
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("---- use NHWC format")
        except RuntimeError as e:
            print("---- use normal format")
            print("failed to enable NHWC: ", e)
    if args.jit:
        try:
            model = torch.jit.trace(model, (inputs, input_lengths), check_trace=False, strict=False)
            print("---- JIT trace enable.")
        except (RuntimeError, TypeError) as e:
            print("---- JIT trace disable.")
            print("failed to use PyTorch jit mode due to: ", e)
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0

    # Forward propagate
    if args.profile and args.device == "xpu":
        for i in range(args.num_iter + args.num_warmup):
            inputs = torch.rand(batch_size, sequence_length, dim)
            input_lengths = torch.LongTensor([12345, 12300, 12000])
            elapsed = time.time()
            inputs = inputs.to(args.device)
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                outputs, output_lengths = model(inputs, input_lengths)
            torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == int((args.num_iter + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                inputs = torch.rand(batch_size, sequence_length, dim)
                input_lengths = torch.LongTensor([12345, 12300, 12000])
                elapsed = time.time()
                inputs = inputs.to(args.device)
                with torch.jit.fuser(fuser_mode):
                    outputs, output_lengths = model(inputs, input_lengths)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                inputs = torch.rand(batch_size, sequence_length, dim)
                input_lengths = torch.LongTensor([12345, 12300, 12000])
                elapsed = time.time()
                inputs = inputs.to(args.device)
                outputs, output_lengths = model(inputs, input_lengths)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i in range(args.num_iter + args.num_warmup):
            inputs = torch.rand(batch_size, sequence_length, dim)
            input_lengths = torch.LongTensor([12345, 12300, 12000])
            elapsed = time.time()
            inputs = inputs.to(args.device)
            with torch.jit.fuser(fuser_mode):
                outputs, output_lengths = model(inputs, input_lengths)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i in range(args.num_iter + args.num_warmup):
            inputs = torch.rand(batch_size, sequence_length, dim)
            input_lengths = torch.LongTensor([12345, 12300, 12000])
            elapsed = time.time()
            inputs = inputs.to(args.device)
            outputs, output_lengths = model(inputs, input_lengths)
            if args.device == "xpu":
                torch.xpu.synchronize()
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

    with torch.no_grad():
        model.eval()
        if args.device == "xpu":
            datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
            model = torch.xpu.optimize(model=model, dtype=datatype)
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                inference(args)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                inference(args)
        else:
            print("---- no autocast")
            inference(args)

if __name__ == "__main__":
    main()

