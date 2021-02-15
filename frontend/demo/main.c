#include "common.h"
#include "graph.h"
#include "tensors.h"

int main(void)
{
    printf("----------->\n\r");
    printf("start test!!!\r\n");
    printf("----------->\r\n");

    int pos = calc_nn(Conv2D1_input, 28, 28);

	printf("Bacon It looks like the number: %d\n", pos);

    return 0;
}


