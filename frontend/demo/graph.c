#include "graph.h"
#include "tensors.h"
#include "ops.h"

const unsigned buffer_size = 2304;
const unsigned buffer_num = 2;
static def_type buffer[2304 * 2];
static unsigned buffer_pos = 0;
#define GET_BUFFER_ADDR(addr) \
    do {\
        addr = &buffer[buffer_pos * buffer_size];\
        buffer_pos = (buffer_pos + 1) % buffer_num;\
    } while(0)

int calc_nn(def_type *input_tensor,
            unsigned H,
            unsigned W)
{
    def_type *in_ts = (def_type *)input_tensor;
    unsigned OW, OH, ICH, OCH, KS;
    KS = 3;
    OW = W - KS + 1;
    OH = H - KS + 1;
    ICH = 1;
    OCH = 2;
    def_type *out; // = buffer;
    GET_BUFFER_ADDR(out);
    conv_2d(in_ts, out, Conv2D1, Conv2D1_bias, W, H, ICH, OCH, KS);

    //限制幅度
    W = OW;
    H = OH;
    relu(out, W, H, OCH);

    //2维的卷积
    in_ts = out;
    KS = 3;
    OW = W - KS + 1;
    OH = H - KS + 1;
    ICH = 2;
    OCH = 4;
    GET_BUFFER_ADDR(out);
    conv_2d(in_ts, out, Conv2D2, Conv2D2_bias, W, H, ICH, OCH, KS);

    //限制幅度
    W = OW;
    H = OH;
    ICH = 4;
    OCH = 4;
    relu(out, W, H, ICH);

    //池化
    in_ts = out;
    OW = W / 2;
    OH = H / 2;
    ICH = 4;
    OCH = 4;
    GET_BUFFER_ADDR(out);
    maxpool_2d(in_ts, out, W, H, ICH, 2);

    //2维卷积
    W = OW;
    H = OH;
    in_ts = out;
    KS = 3;
    OW = W - KS + 1;
    OH = H - KS + 1;
    ICH = 4;
    OCH = 8;
    GET_BUFFER_ADDR(out);
    conv_2d(in_ts, out, Conv2D3, Conv2D3_bias, W, H, ICH, OCH, KS);

    //限制幅度
    W = OW;
    H = OH;
    ICH = 8;
    OCH = 8;
    relu(out, W, H, ICH);

    //2维卷积
    W = OW;
    H = OH;
    in_ts = out;
    KS = 3;
    OW = W - KS + 1;
    OH = H - KS + 1;
    ICH = 8;
    OCH = 16;
    GET_BUFFER_ADDR(out);
    conv_2d(in_ts, out, Conv2D4, Conv2D4_bias, W, H, ICH, OCH, KS);

    //限制幅度
    W = OW;
    H = OH;
    ICH = 16;
    OCH = 16;
    relu(out, W, H, ICH);

    //池化
    in_ts = out;
    OW = W / 2;
    OH = H / 2;
    ICH = 16;
    OCH = 16;
    GET_BUFFER_ADDR(out);

    maxpool_2d(in_ts, out, W, H, ICH, 2);

    //扁平输出
    in_ts = out;
    GET_BUFFER_ADDR(out);
    int index = 0;

    for (int h = 0; h < OH; h++)
    {
        for (int w = 0; w < OW; w++)
        {
            for (int ch = 0; ch < OCH; ch++)
            {
                *(out + index) = *(in_ts + ch * OH * OW + h * OW + w);
                index++;
            }
        }
    }

    OW = OW * OH * OCH;
    OH = 1;

    //矩阵相乘
    in_ts = out;
    W = OW;
    H = OH;
    OH = H;
    OW = 10;
    GET_BUFFER_ADDR(out);
    OCH = 1;
    matmul(in_ts, out, FullConnect, FullConnect_bias, W, H, 10, 256);
    
#if 0 // add for debuging

    for (int niui = 0; niui < 10; niui ++){
        printf("|%f \t- %f \t| = %f \n \r", FullConnect_output[0][niui], out[niui], abs(FullConnect_output[0][niui] - out[niui]));
    }

#endif
    H = OH;
    W = OW;
    in_ts = out;
    GET_BUFFER_ADDR(out);
    OCH = 1;

    soft_max(in_ts, out, OW * OH);
    int pos;
    float val;
    val = out[0];
    pos = 0;
    for (int i = 1; i < 10; i++)
    {
        if (val < out[i])
        {
            val = out[i];
            pos = i;
        }
    }

    return pos;
}