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

#ifndef ARRAYCOLLECTION_H
#define ARRAYCOLLECTION_H
#ifdef __cplusplus
extern "C" {
#endif

static void arraycollection_add(PyArrayObject *ary);

/*
 *===================================================================
 * Get base array that owns the memory address.
 * Return NULL when nobody owns the address.
 */
static PyArrayObject* arraycollection_get(char* addr);

/*
 *===================================================================
 * Remove an base array from the array collection.
 */
static void arraycollection_rm(PyArrayObject *ary);


#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYCOLLECTION_H) */
