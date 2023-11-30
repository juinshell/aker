/*
 * Author: raphael hao
 */

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <iostream>
#include <assert.h>


#define MMAP_FILE "shm-file0"