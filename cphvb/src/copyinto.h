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

#ifndef COPYINTO_H
#define  COPYINTO_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 *===================================================================
 * Copy data from src into dst. The function do nothing when the type
 * is not supported.
 * @dst    Destination array.
 * @src     Source array.
 * @return -1 and set exception on error, 0 on success, 1 when doing
 *         nothing.
 */
static int
PyCphVB_CopyInto(PyArrayObject *dst, PyArrayObject *src);

#ifdef __cplusplus
}
#endif

#endif /* !defined( COPYINTO_H) */
