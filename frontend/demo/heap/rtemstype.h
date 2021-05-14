#ifndef __RTEMS_TYPE_H__
#define __RTEMS_TYPE_H__

#ifndef __cplusplus
#define bool    int
#define true    1
#define false   0
#endif

#define RTEMS_INLINE_ROUTINE static inline

#ifndef NULL
#define NULL    ((void*)0)
#endif

#include <stdint.h>

#endif


