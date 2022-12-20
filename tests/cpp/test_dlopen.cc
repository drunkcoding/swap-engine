// test run of dlopen

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(int argc, char** argv)
{
  void* handle;
  char* error;
  int (*func)(int);

  const char* libname =
      "/opt/tritonserver/backends/pytorch/libtriton_pytorch.so";
  // const char* libname =
  //     "/home/xly/pytorch_backend/build/install/backends/pytorch/libtriton_pytorch.so";

  handle = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    fprintf(stderr, "%s", dlerror());
    exit(1);
  }

  dlerror(); /* Clear any existing error */

  *(void**)(&func) = dlsym(handle, "test_dlopen");
  if ((error = dlerror()) != NULL) {
    fprintf(stderr, "%s", error);
    exit(1);
  }

  printf("test_dlopen() = %d", (*func)(argc));

  dlclose(handle);
  exit(0);
}
