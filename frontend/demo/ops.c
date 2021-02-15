#include "ops.h"

int conv_2d(
        const void *input_tensor,
        void *output_tensor,
        const void *weight_tensor,
        const void *bias_tensor,
        unsigned W,
        unsigned H,
        unsigned IN_CH,
        unsigned OUT_CH,
        unsigned kernel_size)
{

    def_type *in_ts = (def_type*)input_tensor;
    def_type *out_ts = (def_type*)output_tensor;
    def_type *w_ts = (def_type*)weight_tensor;
    def_type *b_ts = (def_type*)bias_tensor;


    unsigned w, h, in_ch;
    unsigned m, n, out_ch;
    unsigned OH, OW;

    OH = H - kernel_size + 1;
    OW = W - kernel_size + 1;

    for (out_ch = 0; out_ch < OUT_CH; out_ch++)
    {
        for (h = 0; h < OH; h++)
        {
            for (w = 0; w < OW; w++)
            {
                def_type res = 0;
                for (in_ch = 0; in_ch < IN_CH; in_ch++)
                {
                    for (m = 0; m < kernel_size; m++)
                    {
                        for (n =0; n < kernel_size; n++)
                        {

                            res += (*(in_ts + in_ch * H * W + (h + m) * W + (w + n))
                                   *
                                   *(w_ts
                                     + in_ch * kernel_size * kernel_size * OUT_CH
                                     + out_ch * kernel_size * kernel_size
                                     + m * kernel_size + n));
                        }
                    }
                }
                res  += *(b_ts + out_ch);
                *(out_ts + out_ch * OW * OH + h * OW + w) = res;
            }
        }
    }

    return 0;
}

int maxpool_2d(const void *input_tensor,
               void *output_tensor,
               unsigned W,
               unsigned H,
               unsigned CH,
               unsigned pool_size)
{
    def_type *in_ts = (def_type*)input_tensor;
    def_type *out_ts = (def_type*)output_tensor;

    unsigned w, h, ch;
    unsigned m, n;
    unsigned OH, OW;

    OH = H / pool_size;
    OW = W / pool_size;


    for (ch = 0; ch < CH; ch++)
    {
        for (m = 0; m < OH; m++)
        {
            for (n = 0; n < OW; n++)
            {
                *(out_ts + ch * OH * OW + m * OW + n) = (def_type)0;
            }
        }
    }

    for (ch = 0; ch < CH; ch++)
    {
        for (h = 0; h < OH * pool_size; h++)
        {
            for (w = 0; w < OW * pool_size; w++)
            {
                def_type *res;
                def_type *val;
                m = h / pool_size;
                n = w / pool_size;

                res = out_ts + ch * OH * OW + m * OW + n;
                val = in_ts + ch * H * W + h * W + w;
                if (*res < *val)
                {
                    *res = *val;
                }
            }
        }
    }
    return 0;
}

int matmul(const void *input_tensor,
           void *output_tensor,
           const void *full_connect,
           const void *full_connect_bias,
           unsigned IN_W,
           unsigned IN_H,
           unsigned FC_W,
           unsigned FC_H)
{
    def_type *in_ts = (def_type*)input_tensor;
    def_type *out_ts = (def_type*)output_tensor;
    def_type *mul_ts = (def_type*)full_connect;
    def_type *b_ts = (def_type*)full_connect_bias;



    unsigned i, j, k;
    unsigned OUT_H = IN_H;
    unsigned OUT_W = FC_W;


    if (IN_W != FC_H)
    {
        printf("error.\n");
        return 1;
    }


    for (i = 0; i < IN_H; i++)
    {
        for (j = 0; j < FC_W; j++)
        {
            def_type pd = 0;
            for (k = 0; k < IN_W; k++)
            {

                pd += *(in_ts + i * IN_W + k) * *(mul_ts + k * FC_W + j);
            }
            *(out_ts + i * FC_W + j) = pd + *(b_ts + i);
        }
    }

    return 0;
}



int soft_max(const void *input_tensor,
             void *output_tensor,
             unsigned inputSize)
{
    def_type *in_ts = (def_type*)input_tensor;
    def_type *out_ts = (def_type*)output_tensor;
#if 0
    def_type sum = (def_type)0;

    for(int i = 0; i < inputSize; i++){
        out_ts[i] = expf((double)in_ts[i]);
        sum += out_ts[i];
    }
    for(int i = 0; i < inputSize; i++){
        out_ts[i] /= sum;
    }
#else

    def_type m = -INFINITY;
    for (int i = 0; i < inputSize; i++) {
      if (in_ts[i] > m) {
        m = in_ts[i];
      }
    }

    def_type sum = 0.0;
    for (int i = 0; i < inputSize; i++) {
      sum += expf(in_ts[i] - m);
    }

    def_type offset = m + logf(sum);
    for (int i = 0; i < inputSize; i++) {
      out_ts [i] = expf(in_ts[i] - offset);
    }

#endif

    return 0;
}



int relu(def_type *inout_tensor,
         unsigned H,
         unsigned W,
         unsigned CH)
{
    def_type *io_ts = (def_type*)inout_tensor;

#if 0
    for (int ch = 0; ch < CH; ch++)
    {
        for (int i = 0; i < H; i++)
        {
            for(int j = 0; j < W; j++)
            {
                def_type val = *(io_ts + ch * H * W + i * W + j);
                if (val < (def_type)0)
                    *(io_ts + ch * H * W + i * W + j) = (def_type)0;
            }
        }
    }
#else
    for (int i = 0; i < H*W*CH; i++)
    {
        def_type val = *(io_ts + i);
        if (val < (def_type)0)
            *(io_ts + i) = (def_type)0;
    }
#endif

    return 0;
}