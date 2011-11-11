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

PyArrayObject *ary_root = NULL;//The root of the base array collection.

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewBaseArray(PyArrayObject *ary);

/*
 *===================================================================
 * Delete array view.
 * When it is the last view of the base array, the base array is de-
 * allocated.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_DelViewArray(PyArrayObject *array);

/*
 *===================================================================
 * Indicate that cphVB should handle the array.
 * @array The array cphVB should handle.
 * @transfer_data Whether data should be transferred from NumPy to
 *                cphVB address space.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_HandleArray(PyArrayObject *array, int transfer_data);

#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYOBJECT_H) */
