/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include "types.h"

cphvb_error cphvb_set_constant(PyArrayObject* ary, cphvb_constant* constant, cphvb_type* type)
{
    *type = type_py2cph[PyArray_TYPE(ary)];  
    switch (PyArray_TYPE(ary))
    {
    case NPY_BOOL:
        constant->bool8 = *(npy_bool*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_BYTE:
        constant->int8 = *(npy_byte*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_UBYTE:
        constant->uint8 = *(npy_ubyte*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_SHORT:
        constant->int16 = *(npy_short*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_USHORT:
        constant->uint16 = *(npy_ushort*)ary->data;
        return CPHVB_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_LONG:
#endif
    case NPY_INT:
        constant->int32 = *(npy_int*)ary->data;
        return CPHVB_SUCCESS;
#if NPY_BITSOF_LONG == 32
    case NPY_ULONG:
#endif
    case NPY_UINT:
        constant->uint32 = *(npy_uint*)ary->data;
        return CPHVB_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_LONG:
#endif
    case NPY_LONGLONG:
        constant->int64 = *(npy_longlong*)ary->data;
        return CPHVB_SUCCESS;
#if NPY_BITSOF_LONG == 64
    case NPY_ULONG:
#endif
    case NPY_ULONGLONG:
        constant->uint64 = *(npy_ulonglong*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_FLOAT:
        constant->float32 = *(npy_float*)ary->data;
        return CPHVB_SUCCESS;
    case NPY_DOUBLE:
        constant->float64 = *(npy_double*)ary->data;
        return CPHVB_SUCCESS;
    default:
        return CPHVB_ERROR;
            
    }
}
