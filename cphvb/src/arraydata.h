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

#ifndef ARRAYDATA_H
#define ARRAYDATA_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 *===================================================================
 * Allocate cphVB-compatible memory.
 * @array  The array that should own the memory.
 * @size   The size of the memory allocation (in bytes).
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_MallocArray(PyArrayObject *ary, cphvb_intp size);

/*
 *===================================================================
 * Free cphVB-compatible memory.
 * @array The array that should own the memory.
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_MfreeArray(PyArrayObject *ary);

/*
 *===================================================================
 * Initialization of the Array Data Protection.
 */
int arydat_init(void);

/*
 *===================================================================
 * Finalization of the Array Data Protection.
 */
int arydat_finalize(void);


#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYDATA_H) */
