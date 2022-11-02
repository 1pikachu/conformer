python setup.py install

# notice: the script haven`t include calculate the loss
python inference.py --device cuda --jit --nv_fuser --precision float16 --profile
