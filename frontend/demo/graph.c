#include "graph.h"
#include "tensors.h"
#include "ops.h"
#include "usermem.h"

#define UAISS_PLAYGROUND_SIZE (4 * 2304 * 2 + 128)
unsigned char uaiss_playground[UAISS_PLAYGROUND_SIZE];

#define ASSERT(a)   \
    do {\
        if (!(a))\
        {\
            printf("ERROR:%d\n", __LINE__);\
        }\
    } while (0)

//From libgcc/memcpy.c
void *
memcpy (void *dest, const void *src, size_t len)
{
  char *d = dest;
  const char *s = src;
  while (len--)
    *d++ = *s++;
  return dest;
}

//Only support 2d tensor input
int calc_nn(def_type *input_tensor, def_type *output_tensor)
{   
    // global initialization
    def_type *in_ts, *out;
    userMallocInitialize(uaiss_playground, UAISS_PLAYGROUND_SIZE, 0);

    // parameter init
    unsigned W, H, ICH;

    // Initiation for convelution 2D layer
    unsigned OCH, KS;

    //Initialization for maxpool2d
    unsigned PS;

    //Initialization for matmul
    unsigned FW, FH;

    //-----------------------------------------

    // CONV_2D_0
    in_ts = (def_type *) input_tensor; //<<input_size>>
    out = (def_type *) userMalloc(1 * 2 * 26 * 26 * sizeof(def_type));
    H = 28; // <<inputWidth>>
    W = 28; // <<inputHeight>>
    ICH = 1; // <<inputChannel>>

    KS = 3; // <<kernelSize>>
    OCH = 2; // <<outputChannel>>

    fused_conv_2d_relu(in_ts, out, Conv2D1, Conv2D1_bias, W, H, ICH, OCH, KS);
    printf("--------------->>\n\r");
// Conv2D1_output
#if 1 // added for debuging
    for (int i = 0; i < 2; i ++){
        for (int j = 0; j < 26; j ++){
            for (int k = 0; k < 26; k ++){
                if (fabs(Conv2D1_output[0][i][j][k] - *(out + i * 26 * 26 + j * 26 + k)) > 0.01){
                    printf("Conv2D1_output Index (%d, %d, %d): |%f \t- %f \t| = %f \n \r", 
                        i, j, k,
                        Conv2D1_output[0][i][j][k], 
                        *(out + i * 26 * 26 + j * 26+k), 
                        fabs(Conv2D1_output[0][i][j][k] - *(out + i * 26 * 26 + j * 26+k))
                    );
                };
            }
        }
    }
    //tests for heap
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //CONV_2D_1
    in_ts = out;
    out = (def_type *) userMalloc(1 * 4 * 24 * 24 * sizeof(def_type));

    W = 26; // <<inputWidth>>
    H = 26; // <<inputHeight>>
    ICH = 2; // <<inputChannel>>

    KS = 3;// <<kernelSize>>
    OCH = 4; // <<outputChannel>>

    fused_conv_2d_relu(in_ts, out, Conv2D2, Conv2D2_bias, W, H, ICH, OCH, KS);

//Conv2D2_output
#if 1 // added for debuging
    for (int i = 0; i < 4; i ++){
        for (int j = 0; j < 24; j ++){
            for (int k = 0; k < 24; k ++){
                if (fabs(Conv2D2_output[0][i][j][k] - *(out + i * 24 * 24 + j * 24 + k)) > 0.01){
                    printf("Conv2D2_output Index (%d, %d, %d): |%f \t- %f \t| = %lf \n \r", 
                        i, j, k,
                        Conv2D2_output[0][i][j][k], 
                        *(out + i * 24 * 24 + j * 24 + k), 
                        fabs(Conv2D2_output[0][i][j][k] - *(out + i * 24 * 24 + j * 24 + k))
                    );
                };
            }
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //MAX_POOL_2D_2
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 4 * 12 * 12 * sizeof(def_type));

    W = 24;
    H = 24;
    ICH = 4; // <<inputChannel>>
    PS = 2; // <<PoolSize>>
    maxpool_2d(in_ts, out, W, H, ICH, PS);

//MAX_POOL_2D_2
#if 1 // added for debuging
    for (int i = 0; i < 4; i ++){
        for (int j = 0; j < 12; j ++){
            for (int k = 0; k < 12; k ++){
                if (fabs(Maxpool1_output[0][i][j][k] - *(out + i * 12 * 12 + j * 12 + k)) > 0.01){
                    printf("MAX_POOL_2D_2 Index (%d, %d, %d): \t |%f \t- %f \t| = %lf \n \r", 
                        i, j, k,
                        Maxpool1_output[0][i][j][k], 
                        *(out + i * 12 * 12 + j * 12 + k), 
                        fabs(Maxpool1_output[0][i][j][k] - *(out + i * 12 * 12 + j * 12 + k))
                    );
                };
            }
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif
    //CONV_2D_3
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 8 * 10 * 10 * sizeof(def_type));

    W = 12; // <<inputWidth>>
    H = 12; // <<inputHeight>>
    ICH = 4; // <<inputChannel>>
    
    KS = 3; // <<kernelSize>>
    OCH = 8; // <<outputChannel>>
    fused_conv_2d_relu(in_ts, out, Conv2D3, Conv2D3_bias, W, H, ICH, OCH, KS);

//CONV_2D_3
#if 1 // added for debuging
    for (int i = 0; i < 8; i ++){
        for (int j = 0; j < 10; j ++){
            for (int k = 0; k < 10; k ++){
                if (fabs(Conv2D3_output[0][i][j][k] - *(out + i * 10 * 10 + j * 10 + k)) > 0.01){
                    printf("CONV_2D_3 Index (%d, %d, %d): |%f \t- %f \t| = %lf  \n \r", 
                        i, j, k,
                        Conv2D3_output[0][i][j][k], 
                        *(out + i * 10 * 10 + j * 10 + k), 
                        fabs(Conv2D3_output[0][i][j][k] - *(out + i * 10 * 10 + j * 10 + k))
                    );
                };
            }
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //CONV_2D_4
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 16 * 8 * 8 * sizeof(def_type));

    W = 10;// <<inputWidth>>
    H = 10;// <<inputHeight>>
    ICH = 8;// <<inputChannel>>
    
    KS = 3;// <<kernelSize>>
    OCH = 16;// <<outputChannel>>
    fused_conv_2d_relu(in_ts, out, Conv2D4, Conv2D4_bias, W, H, ICH, OCH, KS);

//CONV_2D_4
#if 1 // added for debuging
    for (int i = 0; i < 16; i ++){
        for (int j = 0; j < 8; j ++){
            for (int k = 0; k < 8; k ++){
                if (fabs(Conv2D4_output[0][i][j][k] - *(out + i * 8 * 8 + j * 8 + k)) > 0.1){
                    printf("CONV_2D_4 Index (%d, %d, %d): |%f \t- %f \t| = %lf  \n \r", 
                        i, j, k,
                        Conv2D4_output[0][i][j][k], 
                        *(out + i * 8 * 8 + j * 8 + k), 
                        fabs(Conv2D4_output[0][i][j][k] - *(out + i * 8 * 8 + j * 8 + k))
                    );
                };
            }
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //MAX_POOL_2D_5
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 16 * 4 * 4 * sizeof(def_type));

    W = 8;// <<inputWidth>>
    H = 8;// <<inputHeight>>
    ICH = 16;// <<inputChannel>>
    OCH = 16;// <<outputChannel>>

    maxpool_2d(in_ts, out, W, H, ICH, 2);

//MAX_POOL_2D_5
#if 1 // added for debuging
    for (int i = 0; i < 16; i ++){
        for (int j = 0; j < 4; j ++){
            for (int k = 0; k < 4; k ++){
                if (fabs(Maxpool2_output[0][i][j][k] - *(out + i * 4 * 4 + j * 4 + k)) > 0.1){
                    printf("MAX_POOL_2D_5 Index (%d, %d, %d): \t |%f \t- %f \t| = %lf \n \r", 
                        i, j, k,
                        Maxpool2_output[0][i][j][k], 
                        *(out + i * 4 * 4 + j * 4 + k), 
                        fabs(Maxpool2_output[0][i][j][k] - *(out + i * 4 * 4 + j * 4 + k))
                    );
                };
            }
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //RESHAPE_6
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 256 * sizeof(def_type));

    H = 4;// <<inputHeight>>
    W = 4;// <<inputWidth>>
    ICH = 16;// <<inputChannel>>
    flatten(in_ts, out, H, W, ICH);

// Flatten
#if 1 // add for debuging
    for (int i = 0; i < 256; i ++){
        if (fabs(Flatten_output[0][i] - out[i]) > 0.1){
            printf("Flatten index %d : |%f \t- %f \t| = %f \n \r", 
                    i, Flatten_output[0][i], out[i], 
                    fabs(Flatten_output[0][i] - out[i]));
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //FULLY_CONNECTED_7
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 10 * sizeof(def_type));

    W = 256;// <<inputWidth>>
    H = 1;// <<inputHeight>>
    FW = 10;// <<MatmulFilterWidth>>
    FH = 256;// <<MatmulFilterHeight>>
    matmul(in_ts, out, FullConnect, FullConnect_bias, W, H, FW, FH);
    
#if 1 // add for debuging
    for (int i = 0; i < 10; i ++){
        if (fabs(FullConnect_output[0][i] - out[i]) > 0.5){
            printf("|%f \t- %f \t| = %f \n \r", FullConnect_output[0][i], out[i], fabs(FullConnect_output[0][i] - out[i]));
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    // SOFTMAX_8
    userFree(in_ts);
    in_ts = out;
    out = (def_type *) userMalloc(1 * 10 * sizeof(def_type));
    
    H = 1;// <<inputHeight>>
    W = 10;// <<inputWidth>>

    soft_max(in_ts, out, W * H);

#if 1 // add for debuging
    for (int i = 0; i < 10; i ++){
        if (fabs(SoftMax_output[0][i] - out[i]) > 0.01){
            printf("|%f \t- %f \t| = %f \n \r", SoftMax_output[0][i], out[i], fabs(SoftMax_output[0][i] - out[i]));
        }
    }
    ASSERT(in_ts != NULL);
    ASSERT(out != NULL);
    heapwalk();
#endif

    //*output_tensor = *out;
    memcpy(output_tensor, out, sizeof(*out) * 10); // <<OutputTensor size>>

    return 0;
}