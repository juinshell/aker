/*
 * Author: raphael hao
 */

#include "switch.h"
#include "cuda_ipc.h"

int main(int argc, char const *argv[])
{
  int fd = shm_open(MMAP_FILE, O_CREAT | O_RDWR, 0666);
  ftruncate(fd, sizeof(pthread_barrier_t));

  pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  pthread_barrierattr_t barrier_attr;
  pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(shared_barrier, &barrier_attr, 2);
  printf("IPC initialized\n");
  if (argc == 2)
  {
    pthread_barrier_wait(shared_barrier);
    pthread_barrier_wait(shared_barrier);
  }
  return 0;
}
