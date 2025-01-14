// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// This WrapFunc.cpp file is ONLY included in builds where it is needed
//
// We want our library to be loadable into as many Linux distributions as possible.  It seems that on Linux
// the C runtime libraries (libm.so and libc.so) can't be statically linked, or at least it's not advisable
// we therefore want to ensure that we only use older versions of the functions in GLIBC
//
// if we run the command: objdump -T lib_ebm_native_linux_x64.so | grep GLIBC_
// we see the following entry:
// 0000000000000000      DF *UND*  0000000000000000  GLIBC_2.14  memcpy
// the version 2.14 for memcpy has been a problem for some of our users on CentOS
// unfortunately, it seems that this function might be called by the static versions of libgcc OR libstdc++ 
// [verify this] which we're statically linking in (also for compatibility reasons)
// so we need to write our own memcpy.  The easiest thing to do is to call memmove which has the same functionality
// but is safer.  Pretty much all of the places that we call memcpy ourselves, we'd probably benefit by re-writing
// into loops that don't call library functions since we know the exact amount of memory at compile time (thanks to templates)
// so we could straight copy the memory without a loop.
// 
// It's possible to get a list of function versions available to link to with the following command:
// objdump -T /lib/x86_64-linux-gnu/libc.so.6    OR   objdump -T /lib/x86_64-linux-gnu/libm.so.6
//
// we also need to add the following flags to the g++ command (in addition to including this cpp file) : -Wl,--wrap=memcpy
// 
// More info:
// https://stackoverflow.com/questions/8823267/linking-against-older-symbol-version-in-a-so-file
// https://stackoverflow.com/questions/36461555/is-it-possible-to-statically-link-libstdc-and-wrap-memcpy
//

#include <string.h>
#include <math.h>

#if defined(__x86_64__)
// 64 bit x64

__asm__(".symver memcpy, memcpy@GLIBC_2.2.5");
__asm__(".symver exp, exp@GLIBC_2.2.5");
__asm__(".symver log, log@GLIBC_2.2.5");

extern "C" {
   void * __wrap_memcpy(void * dest, const void * src, size_t n) {
      return memcpy(dest, src, n);
   }
   double __wrap_exp(double x) {
      return exp(x);
   }
   double __wrap_log(double x) {
      return log(x);
   }
}

#elif defined(__i386__)
// 32 bit x86

// no GLIBC substitutions

#else
#error unrecognized GCC architecture
#endif

