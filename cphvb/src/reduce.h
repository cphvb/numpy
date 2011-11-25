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

#ifndef REDUCE_H
#define REDUCE_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 *===================================================================
 * Initiate PyCphVB_Reduce().
 * @return -1 and set exception on error, 0 on success.
 */
static int
reduce_init(void);

/*
 *===================================================================
 * Execute an reduce. The function do nothing when the opcode and/or
 * type is not supported.
 * @ufunc  The ufunc object to reduce with.
 * @in     Input Array.
 * @put    Output Array.
 * @axis   The Axis to reduce.
 * @return -1 and set exception on error, 0 on success, 1 when doing
 *         nothing.
 */
static int
PyCphVB_Reduce(PyUFuncObject *ufunc, PyArrayObject *in,
               PyArrayObject *out, int axis);

#ifdef __cplusplus
}
#endif

#endif /* !defined(REDUCE_H) */
