/*
   SipHash reference C implementation

   Copyright (c) 2012-2014 Jean-Philippe Aumasson
   <jeanphilippe.aumasson@gmail.com>
   Copyright (c) 2012-2014 Daniel J. Bernstein <djb@cr.yp.to>

   To the extent possible under law, the author(s) have dedicated all copyright
   and related and neighboring rights to this software to the public domain
   worldwide. This software is distributed without any warranty.

   You should have received a copy of the CC0 Public Domain Dedication along
   with
   this software. If not, see
   <http://creativecommons.org/publicdomain/zero/1.0/>.
 */
#include <stdint.h>

#ifndef SIPHASH24_H
#define SIPHASH24_H
#if defined(__cplusplus)
extern "C" {
#endif

extern int siphash(uint8_t *out, const uint8_t *in, uint64_t inlen, const uint8_t *k);

#if defined(__cplusplus)
}
#endif
#endif
