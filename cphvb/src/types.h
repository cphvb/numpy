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

#ifndef TYPES_H
#define TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb.h>
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


/*===================================================================
 *
 * The data type conversion to and from NumPy and cphVB.
 * Private.
 */
const cphvb_type const type_py2cph[] =
{
    [NPY_BOOL]   = CPHVB_BOOL,
    [NPY_BYTE]   = CPHVB_INT8,
    [NPY_UBYTE]  = CPHVB_UINT8,
    [NPY_SHORT]  = CPHVB_INT16,
    [NPY_USHORT] = CPHVB_UINT16,
    [NPY_INT]    = CPHVB_INT32,
    [NPY_UINT]   = CPHVB_UINT32,
    #if NPY_BITSOF_LONG == 32
        [NPY_LONG]  = CPHVB_INT32,
        [NPY_ULONG] = CPHVB_UINT32,
    #else
        [NPY_LONG]  = CPHVB_INT64,
        [NPY_ULONG] = CPHVB_UINT64,
    #endif
    [NPY_LONGLONG]    = CPHVB_INT64,
    [NPY_ULONGLONG]   = CPHVB_UINT64,
    [NPY_FLOAT]       = CPHVB_FLOAT32,
    [NPY_DOUBLE]      = CPHVB_FLOAT64,
    [NPY_LONGDOUBLE]  = CPHVB_UNKNOWN,
    [NPY_CFLOAT]      = CPHVB_UNKNOWN,
    [NPY_CDOUBLE]     = CPHVB_UNKNOWN,
    [NPY_CLONGDOUBLE] = CPHVB_UNKNOWN,
    [NPY_OBJECT]      = CPHVB_UNKNOWN,
    [NPY_STRING]      = CPHVB_UNKNOWN,
    [NPY_UNICODE]     = CPHVB_UNKNOWN,
    [NPY_VOID]        = CPHVB_UNKNOWN,
    [NPY_NTYPES]      = CPHVB_UNKNOWN,
    [NPY_NOTYPE]      = CPHVB_UNKNOWN,
    [NPY_CHAR]        = CPHVB_UNKNOWN,
    [NPY_USERDEF]     = CPHVB_UNKNOWN
};
const cphvb_type const type_cph2py[] =
{
    [CPHVB_BOOL]    = NPY_BOOL,
    [CPHVB_INT8]    = NPY_BYTE,
    [CPHVB_UINT8]   = NPY_UBYTE,
    [CPHVB_INT16]   = NPY_SHORT,
    [CPHVB_UINT16]  = NPY_USHORT,
    [CPHVB_INT32]   = NPY_INT,
    [CPHVB_UINT32]  = NPY_UINT,
    #if NPY_BITSOF_LONG == 32
        [CPHVB_INT32]  = NPY_LONG,
        [CPHVB_UINT32] = NPY_ULONG,
    #else
        [CPHVB_INT64]  = NPY_LONG,
        [CPHVB_UINT64] = NPY_ULONG,
    #endif
    [CPHVB_INT64]   = NPY_LONGLONG,
    [CPHVB_UINT64]  = NPY_ULONGLONG,
    [CPHVB_FLOAT32] = NPY_FLOAT,
    [CPHVB_FLOAT64] = NPY_DOUBLE
};


#ifdef __cplusplus
}
#endif

#endif /* !defined(TYPES_H) */
