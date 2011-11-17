/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of DistNumPy <https://github.com/distnumpy>.
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

#ifndef CPHVBNUMPY_H
#define CPHVBNUMPY_H
#ifdef __cplusplus
extern "C" {
#endif

//Only import when compiling cphvbnumpymodule.c
#ifdef CPHVBNUMPY_MODULE
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#endif

/* Retrieval of the cphVB array
 *
 * @obj      The (PyArrayObject *).
 * @return   The (cphvb_array *).
 */
#define PyCphVB_ARRAY(obj) (((PyArrayObject *)(obj))->cphvb_ary)


//Import the API.
#include "cphvbnumpy_api.h"


//Check that the definitions in numpy are in accordance with cphVB.
#if NPY_BITSOF_SHORT != 16
#    error the NPY_BITSOF_INT not 16 bit
#endif
#if NPY_BITSOF_INT != 32
#    error the NPY_BITSOF_INT not 32 bit
#endif
#if NPY_BITSOF_LONG != 32 && NPY_BITSOF_LONG != 64
#    error the NPY_BITSOF_LONG not 32 or 64 bit
#endif
#if NPY_BITSOF_LONGLONG != 64
#    error the NPY_BITSOF_LONGLONG not 64 bit
#endif
#if NPY_BITSOF_FLOAT != 32
#    error the NPY_BITSOF_FLOAT not 32 bit
#endif
#if NPY_BITSOF_FLOAT == 64
#    error the NPY_BITSOF_FLOAT not 64 bit
#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(CPHVBNUMPY_H) */
