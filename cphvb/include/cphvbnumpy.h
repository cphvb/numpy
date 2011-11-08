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
#endif

//Flag indicating that the array want to be handled by cphVB
#define CPHVB_WANT 0x2000

//Easy attribute retrievals.
#define PyDistArray_WANT_CPHVB(m) PyArray_CHKFLAGS(m,CPHVB_WANT)
#define PyDistArray_ARRAY(obj) (((PyArrayObject *)(obj))->cphvb_ary)

//Import the API.
#include "cphvbnumpy_api.h"

#ifdef __cplusplus
}
#endif

#endif /* !defined(CPHVBNUMPY_H) */
