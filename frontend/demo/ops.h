#ifndef _OPS_H
#define _OPS_H

#include <math.h>
#include "common.h"

int conv_2d(
        const void *,
        void *,
        const void *,
        const void *,
        unsigned ,
        unsigned ,
        unsigned ,
        unsigned ,
        unsigned);


int maxpool_2d(const void *,
               void *,
               unsigned ,
               unsigned ,
               unsigned ,
               unsigned );

int matmul(const void *,
           void *,
           const void *,
           const void *,
           unsigned ,
           unsigned ,
           unsigned ,
           unsigned );

int soft_max(const void *,
             void *,
             unsigned );

int relu(def_type *,
         unsigned ,
         unsigned ,
         unsigned );

#endif // _OPS_H