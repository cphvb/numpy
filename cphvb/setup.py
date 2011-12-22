"""
/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of DistNumPy <https://github.com/cphvb>.
 *
 * DistNumPy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DistNumPy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DistNumPy. If not, see <http://www.gnu.org/licenses/>.
 */
"""

from distutils.core import setup, Extension
from os.path import join
import os

def build(build_path):
    cphvb_install_dir = join('..','..')#Default location
    try:
        cphvb_install_dir = os.path.abspath(os.environ['CPHVB_INSTALL_DIR'])
    except KeyError:
        pass

    setup(name='cphVB NumPy Bridge',
          version='1.0',
          ext_modules=[Extension(name='cphvbnumpymodule',
                                 sources=[join('cphvb','src','cphvbnumpymodule.c')],
                                 include_dirs=[join('cphvb','include'),
                                               join('cphvb','private'),
                                               join('numpy','core','include'),
                                               join(build_path, 'numpy','core','include','numpy'),
                                               join(cphvb_install_dir, 'include'),],
                                 extra_compile_args=[],
                                 extra_link_args=['-L%s'%join(cphvb_install_dir,'core'), '-lcphvb'],
                                 depends=[join('cphvb','src','arraydata.c'),
                                          join('cphvb','src','arraydata.h'),
                                          join('cphvb','src','arrayobject.c'),
                                          join('cphvb','src','arrayobject.h'),
                                          join('cphvb','src','batch.c'),
                                          join('cphvb','src','batch.h'),
                                          join('cphvb','src','reduce.c'),
                                          join('cphvb','src','reduce.h'),
                                          join('cphvb','src','random.c'),
                                          join('cphvb','src','random.h'),
                                          join('cphvb','src','types.h'),
                                          join('cphvb','src','ufunc.c'),
                                          join('cphvb','src','ufunc.h'),
                                          join('cphvb','src','copyinto.c'),
                                          join('cphvb','src','copyinto.h'),
                                          join('cphvb','src','vem_interface.c'),
                                          join('cphvb','src','vem_interface.h')]
                                 )],
          )

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
