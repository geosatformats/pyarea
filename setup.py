

from  distutils.core import setup
from pkgutil import walk_packages
from itertools import chain
from fnmatch import fnmatch as wc_match

def find_packages(where, exclude=None):
    if not exclude:
        exclude = ()
    if isinstance(where, str):
        where = (where, )
    ret_list = []
    for name in chain.from_iterable(map(lambda w: (n for _, n, ispkg in w if ispkg), (walk_packages(p) for p in where))):
        if not any(wc_match(name, p) for p in exclude):
            ret_list.append(name)

    return tuple(ret_list)

setup(
    name='pyarea',
    author='Ioan Ferencik',
    author_email='ferencikioan@gmail.com',
    version='0.1',
    packages=['pyarea', 'pyarea.navigation'],
    license='LGPL',
    long_description=open('./README').read(),
    data_files=[('lic', ['./LICENSE'])],
    install_requires= [
        ('numpy>=1.5'),
        'pyproj>=1.8',
        'numexpr',
    ]


)
