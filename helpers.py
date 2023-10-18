# --------------------------------------------------------
#       UTILITY FUNCTIONS
# created on May 19th 2016 by M. Reichmann
# --------------------------------------------------------

import pickle
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from subprocess import call
from threading import Thread
from time import time, sleep

from gtts import gTTS
import numpy as np
from os import makedirs, remove, devnull, stat, getenv, _exit
from os import path as pth
from pytz import timezone, utc
from termcolor import colored
from uncertainties import ufloat
from scipy.optimize import curve_fit
import h5py
from functools import partial, wraps
from queue import Queue
from json import load, loads
from inspect import signature
from types import FunctionType, MethodType
from pathlib import Path

OFF = False
ON = True
DEGREE_SIGN = u'\N{DEGREE SIGN}'
COUNT = 0
Dir = Path(__file__).resolve().parent.parent


def choose(v, default, decider='None', *args, **kwargs):
    use_default = decider is None if decider != 'None' else v is None  # noqa
    if callable(default) and use_default:
        default = default(*args, **kwargs)
    return default if use_default else v(*args, **kwargs) if callable(v) else v


def t_str():
    return datetime.now().strftime('%H:%M:%S')


def warning(msg, prnt=True):
    if prnt:
        print(prepare_msg(msg, 'WARNING', 'yellow'))


def critical(msg):
    print(prepare_msg(msg, 'CRITICAL', 'red'))
    _exit(5)


def prepare_msg(msg, head, color=None, attrs=None, blank_lines=0):
    return '{}\r{} {} --> {}'.format('\n' * blank_lines, colored(head, color, attrs=choose(make_list(attrs), None, attrs)), t_str(), msg)


def info(msg, endl=True, blank_lines=0, color='cyan', prnt=True):
    if prnt:
        print(prepare_msg(msg, 'INFO', color, 'dark', blank_lines), flush=True, end='\n' if endl else ' ')
    return time()


def add_to_info(t, msg='Done', prnt=True):
    if prnt:
        print('{m} ({t:2.2f} s)'.format(m=msg, t=time() - t))


def print_check(reset=False):
    global COUNT
    COUNT = 0 if reset else COUNT
    print('======={}========'.format(COUNT))
    COUNT += 1


def untitle(string):
    s = ''
    for word in string.split(' '):
        if word:
            s += word[0].lower() + word[1:] + ' '
    return s.strip(' ')


def round_down_to(num, val=1):
    return int(num) // val * val


def round_up_to(num, val=1):
    return int(num) // val * val + val


def interpolate_two_points(x1, y1, x2, y2, name=''):
    # f = p1*x + p0
    p1 = (y1 - y2) / (x1 - x2)
    p0 = y1 - x1 * p1
    w = abs(x2 - x1)
    fit_range = np.array(sorted([x1, x2])) + [-w / 3., w / 3.]
    f = TF1('fpol1{}'.format(name), 'pol1', *fit_range)
    f.SetParameters(p0, p1)
    return f


def get_x(x1, x2, y1, y2, y):
    return (x2 - x1) / (y2 - y1) * (y - y1) + x1


def get_y(x1, x2, y1, y2, x):
    return get_x(y1, y2, x1, x2, x)


def interpolate_x(x1, x2, y1, y2, y):
    p1 = get_p1(x1, x2, y1, y2)
    p0 = get_p0(x1, y1, p1)
    return (y - p0) / p1 if p1 else 0


def interpolate_y(x1, x2, y1, y2, x):
    p1 = get_p1(x1, x2, y1, y2)
    p0 = get_p0(x1, y1, p1)
    return p1 * x + p0


def get_p1(x1, x2, y1, y2):
    return (y1 - y2) / (x1 - x2) if x1 != x2 else 0


def get_p0(x1, y1, p1):
    return y1 - x1 * p1


def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    for i, (key, value) in enumerate(odict.items()):
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
    return odict


def make_list(value):
    return np.array([] if value is None else [value], dtype=object).flatten()


def remove_file(*file_path, string=None, warn=True):
    for f in file_path:
        if Path(f).exists():
            warning(f'removing {choose(string, f)}', prnt=warn)
            remove(f)


def ensure_dir(path):
    if not pth.exists(path):
        info('Creating directory: {d}'.format(d=path))
        makedirs(path)
    return path


def file_is_beeing_written(file_path):
    file_size = stat(file_path)
    sleep(4)
    return file_size != stat(file_path)


def make_col_str(col):
    return '{0:2d}'.format(int(col)) if int(col) > 1 else '{0:3.1f}'.format(col)


def print_banner(msg, symbol='~', new_lines=1, color=None, prnt=True):
    if prnt:
        msg = '{} |'.format(msg)
        print(colored('{n}{delim}\n{msg}\n{delim}{n}'.format(delim=len(str(msg)) * symbol, msg=msg, n='\n' * new_lines), color))


def print_small_banner(msg, symbol='-', color=None, prnt=True):
    if prnt:
        print(colored('\n{delim}\n{msg}\n'.format(delim=len(str(msg)) * symbol, msg=msg), color))


def print_elapsed_time(start, what='This', show=True, color=None):
    string = f'Elapsed time for {what}: {get_elapsed_time(start)}'
    print_banner(string, color=color) if show else do_nothing()
    return string


def make_byte_string(v):
    n = int(np.log2(v) // 10) if v else 0
    return '{:1.1f} {}'.format(v / 2 ** (10 * n), ['B', 'kB', 'MB', 'GB'][n])


def get_elapsed_time(start, hrs=False):
    t = str(timedelta(seconds=round(time() - start, 0 if hrs else 2)))
    return t if hrs else t[2:-4]


def conv_log_time(time_str, strg=False):
    t = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=utc).astimezone(timezone('Europe/Zurich'))
    return t.strftime('%b %d, %H:%M:%S') if strg else t


def has_bit(num, bit):
    assert (num >= 0 and type(num) is int), 'num has to be non negative int'
    return bool(num & 1 << bit)


def make_tc_str(tc, long_=True, data=False):
    tc_data = str(tc).split('-')
    sub_string = '-{0}'.format(tc_data[-1]) if len(tc_data) > 1 else ''
    if data:
        return datetime.strptime(tc_data[0], '%Y%m').strftime('psi_%Y_%m')
    elif tc_data[0][0].isdigit():
        return '{tc}{s}'.format(tc=datetime.strptime(tc_data[0], '%Y%m').strftime('%B %Y' if long_ else '%b%y'), s=sub_string)
    else:
        return '{tc}{s}'.format(tc=datetime.strptime(tc_data[0], '%b%y').strftime('%Y%m' if long_ else '%B %Y'), s=sub_string)


def tc2str(tc, short=True):
    tc_str = str(tc).split('-')[0]
    sub_str = '-{}'.format(tc.split('-')[-1]) if '-' in str(tc) else ''
    return '{tc}{s}'.format(tc=datetime.strptime(tc_str, '%Y%m').strftime('%b%y' if short else '%B %Y'), s=sub_str)


def flux2str(rate, prec=1, term=False):
    if is_iter(rate):
        return [flux2str(i) for i in rate]
    unit = f'{"MHz" if rate > 900 else "kHz"}/cm{"²" if term else "^{2}"}'
    return f'{rate / (1000 if rate > 900 else 1):2.{prec if rate > 900 else 0}f} {unit}'


def bias2str(bias, root=False):
    return [bias2rootstr(i) if root else bias2str(i) for i in bias] if is_iter(bias) else f'{bias:+.0f} V'


def bias2rootstr(bias):
    if is_iter(bias) and type(bias) is not str:
        return ', '.join(set([bias2str(i) for i in bias]))
    return f'{bias:+.0f} V'.replace('+-', '#pm').replace('+/-', '#pm').replace('+', '#plus').replace('-', '#minus')


def irr2str(val, fmt='.1f'):
    if '?' in str(val):
        return val
    if val == 0:
        return 'non-irradiated'
    val, power = [float(i) for i in f'{val.n:.1e}'.split('e')]
    return f'{val:{fmt}}#upoint10^{{{power:.0f}}} n/cm^{{2}}'


def rp2str(nr):
    return f'{nr:0>2}' if len(str(nr)) <= 2 else f'{nr:0>4}'


def u2str(v, prec=2):
    return f'{v:.{prec}f}'


def make_ev_str(v):
    n = int(np.log10(v) // 3)
    return f'{v / 10 ** (3 * n):.{2 if n > 1 else 0}f}{["", "k", "M"][n]}'


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def isint(x):
    try:
        return float(x) == int(x)
    except (ValueError, TypeError):
        return False


def cart2pol(x, y):
    return np.array([np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)])


def pol2cart(rho, phi):
    return np.array([rho * np.cos(phi), rho * np.sin(phi)])


def print_table(rows, header=None, footer=None, form=None, prnt=True):
    head, foot = [choose([v], np.zeros((0, len(rows[0]))), v) for v in [header, footer]]
    t = np.concatenate([head, rows, foot]).astype('str')
    col_width = [len(max(t[:, i], key=len)) for i in range(t.shape[1])]
    total_width = sum(col_width) + len(col_width) * 3 + 1
    hline = '{}'.format('~' * total_width)
    lines = []
    form = 'l' * t.shape[1] if form is None else form
    for i, row in enumerate(t):
        if i in [0] + choose([1], [], header) + choose([t.shape[0] - 1], [], footer):
            lines.append(hline)
        lines.append('| {r} |'.format(r=' | '.join(word.ljust(n) if form[j] == 'l' else word.rjust(n) for j, (word, n) in enumerate(zip(row, col_width)))))
    lines.append('{}\n'.format(hline))
    if prnt:
        print('\n'.join(lines))
    return '\n'.join(lines)


def do_pickle(path, func, value=None, redo=False, *args, **kwargs):
    path = Path(path)
    if value is not None:
        with open(path, 'wb') as f:
            pickle.dump(value, f)
        return value
    try:
        if path.exists() and not redo:
            with open(path, 'rb') as f:
                return pickle.load(f)
    except ImportError:
        pass
    ret_val = func(redo=redo, *args, **kwargs) if type(func) in [MethodType, FunctionType] and 'redo' in signature(func).parameters else func(*args, **kwargs)
    with open(path, 'wb') as f:
        pickle.dump(ret_val, f)
    return ret_val


def do_hdf5(path, func, redo=False, *args, **kwargs):
    path = Path(path)
    if path.exists() and redo:
        remove_file(path)
    if path.exists() and not redo:
        return h5py.File(path, 'r')['data']
    else:
        data = func(*args, **kwargs)
        f = h5py.File(path, 'w')
        f.create_dataset('data', data=data)
        return f['data']


def int2roman(integer):
    """ Convert an integer to Roman numerals. """
    if type(integer) != int:
        raise TypeError(f'cannot convert {type(integer).__name__} to roman, integer required')
    if not 0 < integer < 4000:
        raise ValueError('Argument must be between 1 and 3999')
    dic = OrderedDict([(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                       (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')])
    result = ''
    for i, num in dic.items():
        count = int(integer // i)
        result += num * count
        integer -= i * count
    return result


def remove_letters(s):
    return ''.join(filter(str.isdigit, s))


def remove_digits(string):
    return ''.join(x for x in string if not x.isdigit())


def average_list(lst, n):
    return [np.mean(lst[i:i + n]) for i in np.arange(0, len(lst), n)] if n > 1 else lst


def log_bins(n_bins, min_val, max_val):
    width = (np.log10(max_val) - np.log10(min_val)) / float(n_bins)
    return [n_bins, np.array([pow(10, np.log10(min_val) + i * width) for i in range(n_bins + 1)])]


def fit2u(fit, par):
    return ufloat(fit.Parameter(par), fit.ParError(par))


def eff2u(eff):
    return ufloat(eff[0], np.mean(eff[1:])) if eff.shape == (3,) else np.array([eff2u(e) for e in eff])


def eff2str(eff, u='\\percent', f='.2f'):
    return f'\\SIerr{{{eff[0]:{f}}}}{{{eff[2]:{f}}}}{{{eff[1]:{f}}}}{{{u}}}'


def add_err(u, e):
    return u + ufloat(0, e)


def add_perr(u, e):
    return np.array([add_perr(i, e) for i in u]) if is_iter(u) else u * ufloat(1, e)


def is_iter(v):
    try:
        iter(v)
        return True
    except TypeError:
        return False


def time_stamp(dt, off=None):
    t = float(dt.strftime('%s'))
    return t if off is None else t - (off if off > 1 else dt.utcoffset().seconds)


def say(txt, lang='en'):
    tts = gTTS(text=txt, lang=lang)
    tts.save('good.mp3')
    with open(devnull, 'w') as FNULL:
        call(([] if getenv('SSH_TTY') is None else ['DISPLAY=:0']) + ['mpg321', 'good.mp3'], stdout=FNULL)
    remove('good.mp3')


def get_running_time(t):
    now = datetime.fromtimestamp(time() - t) - timedelta(hours=1)
    return now.strftime('%H:%M:%S')


def get_arg(arg, default):
    return default if arg is None else arg


def get_buf(buf, n, dtype=None):
    return np.frombuffer(buf, dtype=buf.typecode, count=n).astype(dtype)


def load_json(name):
    with open(name) as f:
        return load(f)


def poly_area(x, y):
    return .5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def discrete_int(x, y):
    """ assume linear interpolation between the points. """
    cut = x.argsort()
    x, y = x[cut], y[cut]
    dx, dy = np.diff(x), np.diff(y)
    i = dx * y[:-1] + .5 * dx * dy
    return sum(i[np.isfinite(i)])


def kramers_kronig(x, y):
    return 1 + 2 / np.pi * np.array([discrete_int(x, x * y / (x ** 2 - ix ** 2)) for ix in x])


def correlate(l1, l2):
    if len(l1.shape) == 2:
        x, y = l1.flatten(), l2.flatten()
        cut, s = (x > 0) & (y > 0), np.count_nonzero(x)
        return correlate(x[cut], y[cut]) if np.count_nonzero(cut) > .6 * s else 0
    return np.corrcoef(l1, l2)[0][1]


def prep_kw(dic, **default):
    d = deepcopy(dic)
    for kw, value in default.items():
        if kw not in d:
            d[kw] = value
    return d


def make_suffix(ana, *values):
    suf_vals = [ana.get_short_name(suf) if type(suf) is str and suf.startswith('TimeIntegralValues') and ana is not None else suf for suf in values]
    return '_'.join(str(int(val) if isint(val) else val.GetName() if hasattr(val, 'GetName') else val) for val in suf_vals if val is not None)


def prep_suffix(f, args, kwargs, suf_args, field=None):
    def_pars = signature(f).parameters
    names, values = list(def_pars.keys()), [par.default for par in def_pars.values()]
    i_arg = (np.arange(len([n for n in names if n not in ['self', '_redo']])) if suf_args == 'all' else make_list(loads(str(suf_args)))) + 1
    suf_vals = [args[i] if len(args) > i else kwargs[names[i]] if names[i] in kwargs else values[i] for i in i_arg]
    suf_vals += [getattr(args[0], str(field))] if field is not None and hasattr(args[0], field) else []
    return make_suffix(args[0], *suf_vals)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_pickle(*pargs, print_dur=False, low_rate=False, high_rate=False, suf_args='[]', field=None, verbose=False, **pkwargs):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if '_no_save' in kwargs:
                return func(*args, **kwargs)
            run = args[0].Run.get_high_rate_run(high=not low_rate) if low_rate or high_rate else None
            pickle_path = Path(args[0].make_simple_pickle_path(*pargs, **prep_kw(pkwargs, run=run, suf=prep_suffix(func, args, kwargs, suf_args, field))))
            info(f'Pickle path: {pickle_path}', prnt=verbose)
            redo = (kwargs['_redo'] if '_redo' in kwargs else False) or (kwargs['show'] if 'show' in kwargs else False)
            if pickle_path.exists() and not redo:
                return load_pickle(pickle_path)
            prnt = print_dur and (kwargs['prnt'] if 'prnt' in kwargs else True)
            t = (args[0].info if hasattr(args[0], 'info') else info)(f'{args[0].__class__.__name__}: {func.__name__.replace("_", " ")} ...', endl=False, prnt=prnt)
            value = func(*args, **kwargs)
            with open(pickle_path, 'wb') as f:
                pickle.dump(value, f)
            (args[0].add_to_info if hasattr(args[0], 'add_to_info') else add_to_info)(t, prnt=prnt)
            return value
        return wrapper
    return inner


def save_hdf5(*pargs, suf_args='[]', **pkwargs):
    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            file_path = Path(args[0].make_simple_hdf5_path(*pargs, **prep_kw(pkwargs, suf=prep_suffix(f, args, kwargs, suf_args))))
            redo = kwargs['_redo'] if '_redo' in kwargs else False
            if file_path.exists() and not redo:
                return h5py.File(file_path, 'r')['data']
            remove_file(file_path)
            data = f(*args, **kwargs)
            hf = h5py.File(file_path, 'w')
            hf.create_dataset('data', data=data)
            return hf['data']
        return wrapper
    return inner


def print_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        prnt = (kwargs['prnt'] if 'prnt' in kwargs else True)
        t = (args[0].info if hasattr(args[0], 'info') else info)(f'{func.__name__.replace("_", " ")} ...', endl=False, prnt=prnt)
        value = func(*args, **kwargs)
        (args[0].add_to_info if hasattr(args[0], 'add_to_info') else add_to_info)(t, prnt=prnt)
        return value
    return wrapper


def quiet(func):
    @wraps(func)
    def wrapper(analysis, *args, **kwargs):
        ana = analysis.Ana if hasattr(analysis, 'Ana') else analysis
        old = ana.Verbose
        ana.set_verbose(False)
        value = func(analysis, *args, **kwargs)
        ana.set_verbose(old)
        return value
    return wrapper


def gauss(x, scale, mean_, sigma, off=0):
    return scale * np.exp(-.5 * ((x - mean_) / sigma) ** 2) + off


def fit_data(f, y, x=None, p=None):
    x = np.arange(y.shape[0]) if x is None else x
    return curve_fit(f, x, y, p0=p)


def multi_threading(lst, timeout=60 * 60 * 2):
    """ runs several threads in parallel. [lst] must contain tuples of the methods and the arguments as list."""
    t0 = info('Run multithreading on {} tasks ... '.format(len(lst)), endl=False)
    lst = [(f, [], {}) for f in lst] if type(lst[0]) not in [list, tuple, np.ndarray] else lst
    if len(lst[0]) == 2:
        lst = [(f, args, {}) for f, args in lst] if type(lst[0][1]) not in [dict, OrderedDict] else [(f, [], d) for f, d in lst]
    threads = []
    queue = Queue()  # use a queue to get the results
    for f, args, kwargs in lst:
        t = Thread(target=lambda q, a, k: q.put(f(*a, **k)), args=(queue, make_list(args), kwargs))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join(timeout)
    add_to_info(t0)
    return [queue.get() for _ in range(queue.qsize())]


def get_attribute(instance, string):
    if '.' in string:
        s = string.split('.')
        return getattr(getattr(instance, s[0]), s[1])
    return getattr(instance, string)


def parallelise(f, args_list, timeout=60 * 60):
    t = info('Run parallelisation on {} tasks ... '.format(len(args_list)), endl=False)
    pool = Pool(cpu_count())
    workers = [pool.apply_async(f, make_list(args)) for args in args_list]
    results = [worker.get(timeout) for worker in workers]
    add_to_info(t)
    return results


def parallelise_instance(instances, method, args, timeout=60 * 60):
    t = info('Run parallelisation on {} tasks ... '.format(len(args)), endl=False)
    pool = Pool(cpu_count())
    # tasks = [partial(call_it, make_list(instances)[0], method.__name__, *make_list(arg)) for arg in args]
    tasks = [partial(call_it, instance, method.__name__, *make_list(arg)) for instance, arg in zip(instances, args)]
    workers = [pool.apply_async(task) for task in tasks]
    results = [worker.get(timeout) for worker in workers]
    add_to_info(t)
    return results


def call_it(instance, name, *args, **kwargs):
    """indirect caller for instance methods and multiprocessing"""
    return getattr(instance, name)(*args, **kwargs)


def get_input(msg, default='None'):
    txt = input(f'{msg} (press enter for default: {default}): ')
    return txt if txt else default


def plural(word, pluralise=True):
    return f'{word}s' if pluralise else word


def alternate(l0, l1):
    """alternatingly concatenates two lists."""
    l0, l1 = np.array(l0), np.array(l1)
    col_vec = lambda x: np.array([x]).T if len(x.shape) == 1 else x.T
    return np.column_stack([col_vec(l0), col_vec(l1)]).flatten()


def do_nothing():
    pass

