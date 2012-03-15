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

#ifndef ARRAYOBJECT_H
#define ARRAYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#include "cphvbnumpy_types.h"

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * NB: The PyArrayObject must be behaved.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_NewBaseArray(PyArrayObject *ary);

/*
 *===================================================================
 * Create a new of a base array and updates the PyArrayObject.
 * NB: The PyArrayObject must be behaved.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_NewViewArray(PyArrayObject *ary);

/*
 *===================================================================
 * Delete array.
 * It is up to the caller to make sure that no view is deleted
 * before its base.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_DelViewArray(PyArrayObject *array);

/*
 *===================================================================
 * Indicate that cphVB should handle the array and all associated
 * array views and the array base.
 *
 * @array         The array cphVB should handle.
 * @transfer_data Whether data should be transferred from NumPy to
 *                cphVB address space.
 * @return        -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_HandleArray(PyArrayObject *array, int transfer_data);

/*
 *===================================================================
 * Indicate that cphVB should NOT handle the array and all associated
 * array views and the array base.
 *
 * @array         The array cphVB should not handle anymore.
 * @implicitly    Whether the unhandling was invoked implicitly by
 *                the bridge or a explicit user request.
 * @return        -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_UnHandleArray(PyArrayObject *array, int implicitly);

/*
 *===================================================================
 * Easy retrieval of array's base.
 *
 * @array         The array view (or base).
 * @return        The base or NULL on error.
 */
static PyArrayObject *
PyCphVB_BaseArray(PyArrayObject *array);

#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYOBJECT_H) */
