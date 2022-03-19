from os.path import join
import numpy as np


def prepend_path(cfg, path_keys, prefix):
    def prepend_path_recur(cfgd):
        if isinstance(cfgd, dict):
            for k, v in cfgd.items():
                if isinstance(v, str):
                    if k in path_keys:
                        cfgd[k] = join(prefix, cfgd[k])
                elif isinstance(v, (dict, list, tuple)):
                    prepend_path_recur(v)
        elif isinstance(cfgd, (list, tuple)):
            for v in cfgd:
                prepend_path_recur(v)
    prepend_path_recur(cfg._cfg_dict)

def print_stats(var, name='', fmt='%.2f', cvt=lambda x: x):
    var = np.asarray(var)
    
    if name:
        prefix = name + ': '
    else:
        prefix = ''

    if len(var) == 1:
        print(('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        ))
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        print(('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        ))
