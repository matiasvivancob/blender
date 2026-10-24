#include <math.h>
#define ccl_restrict __restrict__
template<int IN, int OUT>
__attribute__((noinline)) void dense_rm(const float *ccl_restrict w, const float *ccl_restrict in, float *ccl_restrict out)
{
  const float *b = w + OUT * IN;
  for (int o = 0; o < OUT; o++) {
    const float *wr = w + o * IN;
    float acc = b[o];
    for (int i = 0; i < IN; i++) acc += wr[i] * in[i];
    out[o] = acc;
  }
}
template void dense_rm<28,32>(const float*,const float*,float*);
