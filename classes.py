import json
from configparser import ConfigParser, NoSectionError, NoOptionError
from functools import wraps

from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar, SimpleProgress, Widget
from uncertainties import ufloat_fromstr

from .helpers import choose, do_nothing, time_stamp, timedelta, datetime, Path, colored, critical, load_json, np


def update_pbar(func):
    @wraps(func)
    def my_func(*args, **kwargs):
        value = func(*args, **kwargs)
        if PBAR is not None and PBAR.PBar is not None and not PBAR.is_finished():
            PBAR.update()
        return value
    return my_func


class PBar(object):
    def __init__(self, start=None, counter=False, t=None):
        self.PBar = None
        self.Widgets = self.init_widgets(counter, t)
        self.Step = 0
        self.N = 0
        self.start(start)

    def __reduce__(self):
        return self.__class__, (None, False, None), (self.Widgets, self.Step, self.N)

    def __setstate__(self, state):
        self.Widgets, self.Step, self.N = state
        if self.N:
            self.PBar = ProgressBar(widgets=self.Widgets, maxval=self.N).start()
            self.update(self.Step) if self.Step > 0 else do_nothing()

    @staticmethod
    def init_widgets(counter, t):
        return ['Progress: ', SimpleProgress('/') if counter else Percentage(), ' ', Bar(marker='>'), ' ', ETA(), ' ', FileTransferSpeed() if t is None else EventSpeed(t)]

    def start(self, n, counter=None, t=None):
        if n is not None:
            self.Step = 0
            self.PBar = ProgressBar(widgets=self.Widgets if t is None and counter is None else self.init_widgets(counter, t), maxval=n).start()
            self.N = n

    def update(self, i=None):
        i = self.Step if i is None else i
        if i >= self.PBar.maxval:
            return
        self.PBar.update(i + 1)
        self.Step += 1
        if i == self.PBar.maxval - 1:
            self.finish()

    def set_last(self):
        if self.PBar:
            self.PBar.currval = self.N
            self.PBar.finished = True

    def finish(self):
        self.PBar.finish()

    def is_finished(self):
        return self.PBar.currval == self.N

    def eta(self, i, h, m, s=0):
        self.PBar.start_time = time_stamp(datetime.now() - timedelta(hours=h, minutes=m, seconds=s))
        self.update(i - 1)


class EventSpeed(Widget):
    """Widget for showing the event speed (useful for slow updates)."""

    def __init__(self, t='s'):
        self.unit = t
        self.factor = {'s': 1, 'min': 60, 'h': 60 * 60}[t]

    def update(self, pbar):
        value = 0
        if pbar.seconds_elapsed > 2e-6 and pbar.currval > 2e-6:
            value = pbar.currval / pbar.seconds_elapsed * self.factor
        return f'{value:4.1f} E/{self.unit}'


class Config(ConfigParser):

    def __init__(self, file_name, section=None, from_json=False, required=False, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.FilePath = Path(file_name)
        if required and not self.FilePath.exists():
            critical(f'{self!r} does not exist!')
        self.read_dict(load_json(file_name)) if from_json else self.read(file_name) if type(file_name) is not list else self.read_file(file_name)
        self.Section = self.check_section(section)

    def __call__(self, section):
        return Config(self.FilePath, section)

    def __repr__(self):
        return f'{self.__class__.__name__}: {Path().joinpath(*self.FilePath.parts[-2:])}' + (f' (section = {self.Section})' if hasattr(self, 'Section') and self.Section else '')

    def options(self, section=None):
        return super().options(choose(section, self.Section))

    def check_section(self, section):
        return section if section is None or section in self else critical(f'No section {section} in {self}')

    def set_section(self, sec):
        self.Section = self.check_section(sec)

    def get_value(self, section, option=None, dtype: type = str, default=None):
        dtype = type(default) if default is not None else dtype
        s, o = (self.Section, section) if option is None else (section, option)
        try:
            if dtype is bool:
                return self.getboolean(s, o)
            v = self.get(s, o)
            return json.loads(v.replace('\'', '\"')) if '[' in v or '{' in v and dtype is not str else dtype(v)
        except (NoOptionError, NoSectionError, ValueError):
            return default

    def get_values(self, section=None):
        return [*self[choose(section, self.Section)].values()]

    def get_list(self, section, option=None, default=None):
        return self.get_value(section, option, list, choose(default, []))

    def get_float(self, section: str, option: str = None) -> float:
        return self.get_value(section, option, float)

    def get_ufloat(self, section, option=None, default=None):
        return ufloat_fromstr(self.get_value(section, option, default=default))

    def set_value(self, value, section, option=None):
        s, o = (self.Section, section) if option is None else (section, option)
        self.set(s, o, value)

    def show(self):
        for key, section in self.items():
            print(colored(f'[{key}]', 'yellow'))
            print('\n'.join(f'  {opt} = {val}' for opt, val in section.items()), '\n')

    def write(self, file_name=None, space_around_delimiters=True):
        with open(choose(file_name, self.FilePath), 'w') as f:
            super(Config, self).write(f, space_around_delimiters)


class NumStr(int):

    D = {'': 1, 'K': 1e3, 'M': 1e6, 'G': 1e6}

    def __new__(cls, s: str | int):
        x = None
        if type(s) is str:
            m = s[-1].upper() if s[-1].upper() in cls.D.keys() else ''
            x = super(NumStr, cls).__new__(cls, int(float(s[:-1] if len(m) > 0 else s) * cls.D[m]))
            x.StringMultiplier = m
            x.String = s
        elif type(s) is int:
            x = super(NumStr, cls).__new__(cls, s)
            x.StringMultiplier = list(cls.D.keys())[int(np.log10(s) // 3)]
            x.String = f'{s / cls.D[x.StringMultiplier]:.1f}{x.StringMultiplier}'
        return x

    def __str__(self):
        return self.String


PBAR = PBar()
