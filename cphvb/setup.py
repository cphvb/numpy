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
    cphvb_install_dir = join('..','..','..','..')#Default location
    try:
        cphvb_install_dir = os.path.abspath(os.environ['CPHVB_INSTALL_DIR'])
    except KeyError:
        pass
    cphvb_install_dir = join(cphvb_install_dir, 'include')

    print "build_path: ", build_path
    setup(name='cphVB NumPy Bridge',
          version='1.0',
          ext_modules=[Extension(name='cphvbnumpymodule',
                                 sources=[join('cphvb','src','cphvbnumpymodule.c')],
                                 include_dirs=[join('cphvb','include'),
                                               join('cphvb','private'),
                                               join('numpy','core','include'),
                                               join(build_path, 'numpy','core','include','numpy'),
                                               cphvb_install_dir,],
                                 extra_compile_args=[],
                                 extra_link_args=['-lcphvb'],
                                 depends=[join('cphvb','src','helpers.c'),
                                          join('cphvb','src','helpers.h'),
                                          join('cphvb','src','array_database.c'),
                                          join('cphvb','src','array_database.h'),
                                          join('cphvb','src','arrayobject.c'),
                                          join('cphvb','src','arrayobject.h'),
                                          join('cphvb','src','dependency_system.c'),
                                          join('cphvb','src','dependency_system.h'),
                                          join('cphvb','src','process_grid.c'),
                                          join('cphvb','src','process_grid.h'),
                                          join('cphvb','src','arraydata.c'),
                                          join('cphvb','src','arraydata.h'),
                                          join('cphvb','src','memory.c'),
                                          join('cphvb','src','memory.h')]
                                 )],
          )

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')