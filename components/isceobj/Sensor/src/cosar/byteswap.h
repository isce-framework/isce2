#ifndef _BYTESWAP_H
#define _BYTESWAP_H

#warning "byteswap.h is an unportable GNU extension!  Don't use!"

static inline unsigned short bswap_16(unsigned short x) {
      return (x>>8) | (x<<8);
}

static inline unsigned int bswap_32(unsigned int x) {
      return (bswap_16(x&0xffff)<<16) | (bswap_16(x>>16));
}

static inline unsigned long long bswap_64(unsigned long long x) {
      return (((unsigned long long)bswap_32(x&0xffffffffull))<<32) | (bswap_32(x>>32));
}

static int is_big_endian() {
  union {
   long l;
   char c[sizeof (long) ];
  } u;
  u.l = 1;
  return( u.c[sizeof(long) - 1] ==  1 ? 1 : -1);
}

#endif
