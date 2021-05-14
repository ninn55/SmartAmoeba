#ifndef __USERMEM_H__
#define __USERMEM_H__



#include <heap/rtemstype.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

void *userRealloc(
  void *ptr,
  size_t size
);

void *userMalloc(
  size_t  size
);

void userMallocInitialize(
  void *heap_begin,
  uintptr_t heap_size,
  size_t sbrk_amount
);

void userFree(
  void *ptr
);

#ifdef __cplusplus
}
#endif

#endif

