#include "common.h"
#include "graph.h"
#include "tensors.h"
#include "ops.h"

int main(void)
{
    printf("----------->\n\r");
    printf("start test!!!\r\n");
    printf("----------->\r\n");

    const int output_size = 10;
    def_type output_buffer[output_size];
    int pos;

    if ( calc_nn((def_type *) Conv2D1_input, output_buffer) != 0 ){
        printf("Run failed!!!");
        return 1;
    }
    
#if 1 // add for debuging
    for (int i = 0; i < 10; i ++)
        printf("%f \n \r", output_buffer[i]);
#endif

    //argmax
    argmax(output_buffer, &pos, output_size);

	printf("Bacon It looks like the number: %d\n", pos);

    return 0;
}
