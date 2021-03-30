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

int fused_conv_2d_relu(
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

int fused_matmul_relu(const void *,
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

int flatten(const void *,
           void *,
           unsigned ,
           unsigned ,
           unsigned);

int argmax(const void *,
           unsigned *,
           unsigned );

int add(const void *,
        const void *,
        void *,
        unsigned );

#endif // _OPS_H