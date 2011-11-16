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

#ifndef UFUNC_H
#define UFUNC_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 *===================================================================
 * Execute an ufunc.
 * @ufunc  The ufunc object.
 * @op     List of operands (inputs before outputs).
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_Ufunc(PyUFuncObject *ufunc, PyArrayObject **op);

#ifdef __cplusplus
}
#endif

#endif /* !defined(UFUNC_H) */
