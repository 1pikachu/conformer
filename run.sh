pyhton setup.py install

python inference.py --device cuda --jit --nv_fuser --precision float16 --profile
