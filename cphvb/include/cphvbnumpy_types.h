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

#ifndef CPHVBNUMPY_TYPES_H
#define CPHVBNUMPY_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb.h>

// Extension to the PyArrayObject
/*
    Pointer to the cphvb array.
    NULL when this is a strictly NumPy array.
    cphvb_array*     cphvb_ary;

    Whether cphvb handles the array or not.
    NB: this is always false when cphvb_ary is NULL.
    int            cphvb_handled;

    Memory protected start address (incl.).
    npy_intp mprotected_start;

    memory protected end address (excl.).
    npy_intp mprotected_end;

    Next array in the array collection.
    PyObjectArray *next;

    Previous array in the array collection.
    PyObjectArray *prev;

    Number of bytes allocated for array data
    NB: this may be more than the array size.
    npy_intp      data_allocated;
*/
#define CPHVBNUMPY_ARRAY                \
    cphvb_array*   cphvb_ary;           \
    int            cphvb_handled;       \
    char*          mprotected_start;    \
    char*          mprotected_end;      \
    PyArrayObject *next;                \
    PyArrayObject *prev;                \
    npy_intp       data_allocated;      \


// Extension to the PyUFuncGenericFunction
/*
    The cphVB opcode that correspond to the ufunc.
    cphvb_opcode   opcode;
*/
#define CPHVBNUMPY_UFUNC                \
    cphvb_opcode   opcode;              \


#ifdef __cplusplus
}
#endif

#endif /* !defined(CPHVBNUMPY_TYPES_H) */
