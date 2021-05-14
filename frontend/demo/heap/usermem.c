
#include <heap/usermem.h>
#include <heap/rtemstype.h>
#include <heap/heap.h>
#include <heap/heapconf.h>
#include <heap/rtemstype.h>
#include <string.h>

static Heap_Control gUser_Malloc_Heap;
Heap_Control  *RTEMS_Malloc_Heap = &gUser_Malloc_Heap;



void heapwalk()
{
  _Heap_Walk(
  RTEMS_Malloc_Heap,
  1,
  1);
  
}
/*!
\brief ��ʼ����ջ��
\param heap_begin �ѵĳ�ʼ����ַ��
\param heap_size �ѵĴ�С��
\param sbrk_amount �ѿռ䲻��������£���Ҫ���ӿռ�Ĵ�С��û��ʹ�á��򵥵Ĵ���0��
\return �ޡ�
\warning 
\arg �˺��������ڵ���������ջ����ǰ���á�
\arg �˺��������̰߳�ȫ�ġ�
*/
void userMallocInitialize(
  void *heap_begin,
  uintptr_t heap_size,
  size_t sbrk_amount
)
{
    memset( heap_begin, 0, heap_size );

    uintptr_t status = _Heap_Initialize(
      RTEMS_Malloc_Heap,
      heap_begin,
      heap_size,
      CPU_HEAP_ALIGNMENT
    );
    while ( status == 0 );
}

/*!
\brief ��ջ�������ڴ档
\param size ��Ҫ�����ڴ�ĳߴ硣
\return �ɹ����������ڴ�ĵ�ַ��ʧ�ܷ���NULL��
\warning 
\arg �˺��������̰߳�ȫ�ġ�
*/
void *userMalloc(
  size_t  size
)
{
  void        *return_this;
  /*
   * Validate the parameters
   */
  if ( !size )
    return (void *) 0;



  /*
   * Try to give a segment in the current heap if there is not
   * enough space then try to grow the heap.
   * If this fails then return a NULL pointer.
   */

  return_this = _Heap_Allocate( RTEMS_Malloc_Heap, size );

  if ( !return_this ) {
#if 0
    if (rtems_malloc_sbrk_helpers)
      return_this = (*rtems_malloc_sbrk_helpers->extend)( size );
#endif
    if ( !return_this ) {
#if 0
      errno = ENOMEM;
#endif
      return (void *) 0;
    }
  }

  return return_this;
}


/*!
\brief �ͷ��ڴ�ռ䡣
\param size ��Ҫ�����ڴ�ĳߴ硣
\return ��
\warning 
\arg �˺��������̰߳�ȫ�ġ�
*/
void userFree(
  void *ptr
)
{
  if ( !ptr )
    return;

  if ( !_Heap_Free( RTEMS_Malloc_Heap, ptr ) ) {
    printk( "Program heap: free of bad pointer %p -- range %p - %p \n",
      ptr,
      RTEMS_Malloc_Heap->area_begin,
      RTEMS_Malloc_Heap->area_end
    );
  }
}


/*!
\brief �ͷ��ڴ�ռ䡣
\param size ��Ҫ�����ڴ�ĳߴ硣
\return �ɹ����������ڴ�ĵ�ַ��
\warning 
\arg �˺��������̰߳�ȫ�ġ�
*/
void *userRealloc(
  void *ptr,
  size_t size
)
{
  uintptr_t old_size;
  char    *new_area;
  uintptr_t resize;

#if 0
  MSBUMP(realloc_calls, 1);
#endif
  /*
   *  Do not attempt to allocate memory if in a critical section or ISR.
   */
#if 0
  if (_System_state_Is_up(_System_state_Get())) {
    if (_Thread_Dispatch_disable_level > 0)
      return (void *) 0;

    if (_ISR_Nest_level > 0)
      return (void *) 0;
  }
#endif

  /*
   * Continue with realloc().
   */
  if ( !ptr )
    return userMalloc( size );

  if ( !size ) {
    userFree( ptr );
    return (void *) 0;
  }

  if ( !_Heap_Size_of_alloc_area(RTEMS_Malloc_Heap, ptr, &old_size) ) {
    return (void *) 0;
  }

  /*
   *  If block boundary integrity checking is enabled, then
   *  we need to account for the boundary memory again.
   */
  resize = size;


  uintptr_t          old_mem_size;
  uintptr_t          avail_mem_size;


  if (HEAP_RESIZE_SUCCESSFUL ==  _Heap_Resize_block( RTEMS_Malloc_Heap, ptr, resize, &old_mem_size, &avail_mem_size ) ) {
    return ptr;
  }

  /*
   *  There used to be a free on this error case but it is wrong to
   *  free the memory per OpenGroup Single UNIX Specification V2
   *  and the C Standard.
   */

  new_area = userMalloc( size );


  if ( !new_area ) {
    return (void *) 0;
  }

  memcpy( new_area, ptr, (size < old_size) ? size : old_size );
  userFree( ptr );

  return new_area;

}
