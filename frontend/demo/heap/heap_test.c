#include "usermem.h"

#define UAISS_PLAYGROUND_SIZE (52304)
unsigned char uaiss_playground[UAISS_PLAYGROUND_SIZE];


#define ASSERT(a)   \
    do {\
        if (!(a))\
        {\
            printf("ERROR:%d\n", __LINE__);\
        }\
    } while (0)

int main(void)
{
    void *p1, *p2, *p3, *p4;
    int a;
    //printf("100. %d\n", sizeof(p1));
    userMallocInitialize(uaiss_playground, UAISS_PLAYGROUND_SIZE, 0);
    //printf("yes.\n");

    p1 = userMalloc(3072);
    ASSERT(p1 != NULL);

    p2 = userMalloc(16384);
    ASSERT(p2 != NULL);

    userFree(p1);
    p1 = NULL;

    p1 = userMalloc(16384);
    ASSERT(p1 != NULL);

    heapwalk();

    p3 = userMalloc(16384);
    ASSERT(p3 != NULL);

    userFree(p1);
    p1 = NULL;

    p1 = userMalloc(16384);
    ASSERT(p1 != NULL);


    heapwalk();



    return 0;
}

