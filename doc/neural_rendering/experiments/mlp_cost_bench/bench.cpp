/* bench.cpp: cost of a neural BSDF decoder vs. analytic BSDFs, kernel-style C++.
 *
 * Part 1 of this file (up to the "Benchmark harness" banner) is written the way
 * Cycles kernel code is written: no STL, no dynamic allocation, fixed-size
 * loops with compile-time bounds, float math only, small functions marked
 * ccl_device_inline (always_inline on GCC/Clang, as in
 * intern/cycles/util/defines.h). Part 2 is the (ordinary C++) harness.
 *
 * Part 1 contents:
 *   1. Analytic reference: GGX microfacet reflection (Smith G, Schlick F) +
 *      Lambert diffuse, layered (diffuse attenuated by the coat Fresnel).
 *      Follows intern/cycles/kernel/closure/bsdf_microfacet.h (bsdf_D,
 *      bsdf_lambda, bsdf_microfacet_eval) and bsdf_diffuse.h.
 *   2. Neural decoder: fixed-size MLP, LeakySmeLU activation, exp() output,
 *      input encoding = (wi, wo, h, shortest-arc difference vector) + latent.
 *   3. Evaluation strategies for the MLP (names as used in the result tables):
 *        rm  : one eval at a time, row-major weights W[out][in] (as specified),
 *              plain loops
 *        im  : one eval at a time, input-major weights W[in][out], plain loops
 *              (compiler vectorizes across the outputs, no reassociation needed)
 *        imv : one eval at a time, input-major weights padded to 8 outputs,
 *              hand-written float8 over outputs, split accumulators (the best
 *              a careful kernel author can do for a single path vertex)
 *        b8n : 8 evals batched (SoA), plain loops, compiler vectorizes across
 *              evals, one output neuron at a time
 *        b8t : same but 8 output neurons accumulated together (register tile)
 *        b8v : 8 evals batched, explicit 8-wide vector type (like Cycles'
 *              float8), register tile of 8 outputs. Upper bound for a
 *              hand-vectorized "wavefront neural stage".
 *
 * All generic math is templated on T = float (scalar) or float8 (AVX2) so the
 * exact same source is timed scalar and vectorized.
 */

#include <math.h>

#if defined(__SSE__)
#  include <immintrin.h>
#endif
#if defined(__AVX2__)
#  define NB_HAVE_FLOAT8 1
#endif

#define ccl_device static inline
#define ccl_device_inline static inline __attribute__((always_inline))
#define ccl_device_forceinline static inline __attribute__((always_inline))
#define ccl_restrict __restrict__

#define M_PI_F 3.14159265358979323846f
#define M_1_PI_F 0.318309886183790671538f
#define M_LN2_F 0.6931471805599453094f

/* Prevents the compiler from vectorizing across loop iterations of the caller
 * (models one scalar eval per path vertex) and from removing dead results. It
 * emits no instructions. */
#if defined(__SSE__)
#  define BENCH_CONSUME(v) __asm__ volatile("" : "+x"(v))
#else
#  define BENCH_CONSUME(v) __asm__ volatile("" : "+r"(v))
#endif

/* ------------------------------------------------------------------------- */
/* Scalar helpers (same semantics as Cycles' util/math.h) and an 8-wide type.  */

ccl_device_inline float sqr(float a)
{
  return a * a;
}
ccl_device_inline float vsqrt(float a)
{
  return sqrtf(a);
}
ccl_device_inline float vabs(float a)
{
  return fabsf(a);
}
ccl_device_inline float vmax(float a, float b)
{
  return fmaxf(a, b);
}
ccl_device_inline float vmin(float a, float b)
{
  return fminf(a, b);
}
/* Cycles' min()/max() (= std::min/std::max, no NaN handling) as used by
 * clamp(); vmin()/vmax() above are fminf/fmaxf, which is what the BSDF code
 * itself calls on Linux. NOTE: GCC compiles scalar (a < b) ? b : a to a
 * compare-and-branch unless -ffinite-math-only is on, so the *activation*
 * below deliberately avoids min/max/clamp (see leaky_smelu). */
ccl_device_inline float kmin(float a, float b)
{
  return (b < a) ? b : a;
}
ccl_device_inline float kmax(float a, float b)
{
  return (a < b) ? b : a;
}
ccl_device_inline float clamp(float a, float lo, float hi)
{
  return kmin(kmax(a, lo), hi);
}
ccl_device_inline float madd(float a, float b, float c)
{
  return a * b + c;
}
/* c > 0 ? a : b */
ccl_device_inline float vselect_pos(float c, float a, float b)
{
  return c > 0.0f ? a : b;
}

#ifdef NB_HAVE_FLOAT8
struct float8 {
  __m256 m;
  float8() = default;
  float8(__m256 v) : m(v) {}
  float8(float s) : m(_mm256_set1_ps(s)) {}
};
ccl_device_inline float8 operator+(float8 a, float8 b) { return _mm256_add_ps(a.m, b.m); }
ccl_device_inline float8 operator-(float8 a, float8 b) { return _mm256_sub_ps(a.m, b.m); }
ccl_device_inline float8 operator*(float8 a, float8 b) { return _mm256_mul_ps(a.m, b.m); }
ccl_device_inline float8 operator/(float8 a, float8 b) { return _mm256_div_ps(a.m, b.m); }
ccl_device_inline float8 operator-(float8 a) { return _mm256_sub_ps(_mm256_setzero_ps(), a.m); }
ccl_device_inline float8 vsqrt(float8 a) { return _mm256_sqrt_ps(a.m); }
ccl_device_inline float8 vabs(float8 a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.m); }
ccl_device_inline float8 vmax(float8 a, float8 b) { return _mm256_max_ps(a.m, b.m); }
ccl_device_inline float8 vmin(float8 a, float8 b) { return _mm256_min_ps(a.m, b.m); }
ccl_device_inline float8 kmax(float8 a, float8 b) { return _mm256_max_ps(a.m, b.m); }
ccl_device_inline float8 kmin(float8 a, float8 b) { return _mm256_min_ps(a.m, b.m); }
ccl_device_inline float8 clamp(float8 a, float8 lo, float8 hi) { return kmin(kmax(a, lo), hi); }
ccl_device_inline float8 madd(float8 a, float8 b, float8 c) { return _mm256_fmadd_ps(a.m, b.m, c.m); }
ccl_device_inline float8 vselect_pos(float8 c, float8 a, float8 b)
{
  return _mm256_blendv_ps(b.m, a.m, _mm256_cmp_ps(c.m, _mm256_setzero_ps(), _CMP_GT_OQ));
}
ccl_device_inline float8 load8(const float *p) { return _mm256_loadu_ps(p); }
ccl_device_inline void store8(float *p, float8 v) { _mm256_storeu_ps(p, v.m); }
#endif

/* fast_exp2f / fast_expf from intern/cycles/util/math_fast.h. */
ccl_device_inline float fast_exp2f(float x)
{
  x = clamp(x, -126.0f, 126.0f);
  const int m = (int)x;
  x -= m;
  x = 1.0f - (1.0f - x); /* crush denormals */
  float r = 1.33336498402e-3f;
  r = madd(x, r, 9.810352697968e-3f);
  r = madd(x, r, 5.551834031939e-2f);
  r = madd(x, r, 0.2401793301105f);
  r = madd(x, r, 0.693144857883f);
  r = madd(x, r, 1.0f);
  return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, r) + ((unsigned)m << 23));
}
ccl_device_inline float fast_expf(const float x)
{
  return fast_exp2f(x / M_LN2_F);
}

#ifdef NB_HAVE_FLOAT8
/* Same algorithm, 8 lanes. */
ccl_device_inline float8 fast_expf(const float8 x0)
{
  float8 x = clamp(x0 / M_LN2_F, -126.0f, 126.0f);
  const __m256i m = _mm256_cvttps_epi32(x.m);
  x = x - float8(_mm256_cvtepi32_ps(m));
  float8 r = 1.33336498402e-3f;
  r = madd(x, r, 9.810352697968e-3f);
  r = madd(x, r, 5.551834031939e-2f);
  r = madd(x, r, 0.2401793301105f);
  r = madd(x, r, 0.693144857883f);
  r = madd(x, r, 1.0f);
  return _mm256_castsi256_ps(
      _mm256_add_epi32(_mm256_castps_si256(r.m), _mm256_slli_epi32(m, 23)));
}
#endif

/* ------------------------------------------------------------------------- */
/* 1. Analytic reference: GGX reflection + Lambert, layered.                  */
/*                                                                            */
/* Local frame: N = +Z. Parameters: roughness (alpha = roughness^2 as in the   */
/* Principled BSDF), specular F0 (RGB), diffuse albedo (RGB).                 */
/*                                                                            */
/* Branch-free (validity applied as a select) so the same source vectorizes.  */
/* Cycles' early-out for cos_NI <= 0 never triggers on our data, so scalar     */
/* timing is unaffected.                                                      */

struct AnalyticParams {
  float rough;
  float f0[3];
  float albedo[3];
  float pad;
};

template<class T> ccl_device_inline T schlick_weight(const T cos_theta)
{
  const T m = clamp(T(1.0f) - cos_theta, 0.0f, 1.0f);
  const T m2 = m * m;
  return m2 * m2 * m;
}

/* bsdf_lambda<GGX>: 0.5 * (sqrt(1 + alpha2 * tan^2) - 1). */
template<class T> ccl_device_inline T ggx_lambda(const T alpha2, const T cos_N)
{
  return 0.5f * (vsqrt(1.0f + alpha2 * vmax(T(1.0f) / (cos_N * cos_N) - 1.0f, 0.0f)) - 1.0f);
}

/* bsdf_D<GGX>. */
template<class T> ccl_device_inline T ggx_D(const T alpha2, const T cos_NH)
{
  const T cos_NH2 = kmin(cos_NH * cos_NH, 1.0f);
  const T one_minus_cos_NH2 = T(1.0f) - cos_NH2;
  const T den = one_minus_cos_NH2 + alpha2 * cos_NH2;
  return alpha2 / (M_PI_F * (den * den));
}

/* WITH_SPEC / WITH_DIFFUSE select the lobes; WITH_PDF additionally computes the
 * pdf the way bsdf_microfacet_eval / bsdf_diffuse_eval do (and mixes the two
 * lobes by their average weight, as closure sampling does).
 * rgb receives BSDF * cos(wo), like Cycles' eval. */
template<class T, bool WITH_SPEC, bool WITH_DIFFUSE, bool WITH_PDF>
ccl_device_inline void analytic_eval(const T wix, const T wiy, const T wiz,
                                     const T wox, const T woy, const T woz,
                                     const T rough,
                                     const T f0_r, const T f0_g, const T f0_b,
                                     const T alb_r, const T alb_g, const T alb_b,
                                     T *ccl_restrict rgb,
                                     T *ccl_restrict pdf)
{
  const T cos_NI = wiz;
  const T cos_NO = woz;

  const T alpha = rough * rough;
  const T alpha2 = alpha * alpha;

  /* Half vector. */
  T Hx = wix + wox, Hy = wiy + woy, Hz = wiz + woz;
  const T len_H = vsqrt(Hx * Hx + Hy * Hy + Hz * Hz);
  const T inv_len_H = vselect_pos(len_H, T(1.0f) / len_H, T(0.0f));
  Hx = Hx * inv_len_H;
  Hy = Hy * inv_len_H;
  Hz = Hz * inv_len_H;
  const T cos_HI = Hx * wix + Hy * wiy + Hz * wiz;
  const T cos_NH = Hz;

  /* Schlick Fresnel with f90 = 1. */
  const T sw = schlick_weight(cos_HI);
  const T F_r = f0_r + (T(1.0f) - f0_r) * sw;
  const T F_g = f0_g + (T(1.0f) - f0_g) * sw;
  const T F_b = f0_b + (T(1.0f) - f0_b) * sw;

  T out_r = 0.0f, out_g = 0.0f, out_b = 0.0f;
  T pdf_spec = 0.0f, pdf_diff = 0.0f;
  T w_spec = 0.0f, w_diff = 0.0f;

  if (WITH_SPEC) {
    const T D = ggx_D(alpha2, cos_NH);
    const T lambdaI = ggx_lambda(alpha2, cos_NI);
    const T lambdaO = ggx_lambda(alpha2, cos_NO);
    const T common = D / cos_NI * 0.25f;
    const T spec = common / (1.0f + lambdaO + lambdaI);
    out_r = out_r + F_r * spec;
    out_g = out_g + F_g * spec;
    out_b = out_b + F_b * spec;
    if (WITH_PDF) {
      pdf_spec = common / (1.0f + lambdaI);
      w_spec = (F_r + F_g + F_b) * (1.0f / 3.0f);
    }
  }

  if (WITH_DIFFUSE) {
    /* Layered: light enters and leaves through the specular coat. Coat
     * reflectance approximated by the average Schlick F at cos_NI / cos_NO. */
    const T f0_avg = (f0_r + f0_g + f0_b) * (1.0f / 3.0f);
    const T Fi = f0_avg + (T(1.0f) - f0_avg) * schlick_weight(cos_NI);
    const T Fo = f0_avg + (T(1.0f) - f0_avg) * schlick_weight(cos_NO);
    const T coat = (T(1.0f) - Fi) * (T(1.0f) - Fo);
    const T cosNO_pi = vmax(cos_NO, 0.0f) * M_1_PI_F;
    out_r = out_r + alb_r * coat * cosNO_pi;
    out_g = out_g + alb_g * coat * cosNO_pi;
    out_b = out_b + alb_b * coat * cosNO_pi;
    if (WITH_PDF) {
      pdf_diff = cosNO_pi;
      w_diff = (alb_r + alb_g + alb_b) * (1.0f / 3.0f) * coat;
    }
  }

  /* Both directions must be in the upper hemisphere (Cycles returns zero otherwise). */
  const T valid = kmin(cos_NI, cos_NO);
  rgb[0] = vselect_pos(valid, out_r, T(0.0f));
  rgb[1] = vselect_pos(valid, out_g, T(0.0f));
  rgb[2] = vselect_pos(valid, out_b, T(0.0f));
  if (WITH_PDF) {
    const T wsum = w_spec + w_diff;
    *pdf = vselect_pos(valid, (w_spec * pdf_spec + w_diff * pdf_diff) / kmax(wsum, 1e-8f), T(0.0f));
  }
}

/* ------------------------------------------------------------------------- */
/* 2. Neural decoder building blocks.                                         */

/* LeakySmeLU: slope eps in the inactive region (x < -beta), identity above
 * (x > beta), quadratic transition for |x| <= beta so the function is C1:
 *   f(x) = eps*x + (1-eps) * ( c(x)^2 / (4beta) + max(x-beta, 0) ),
 *   c(x) = clamp(x+beta, 0, 2beta),
 * which equals (1-eps)/(4beta) x^2 + (1+eps)/2 x + beta(1-eps)/4 for |x| < beta.
 * clamp and max are written with |.| (clamp(t,0,2b) = (|t| - |t-2b|)/2 + b,
 * max(y,0) = (y+|y|)/2): pure arithmetic, so every compiler emits branch-free,
 * vectorizable code even without -ffinite-math-only. */
#define NB_BETA 0.1f
#define NB_EPS 0.01f

template<class T> ccl_device_inline T leaky_smelu(const T x)
{
  const T t = x + NB_BETA;
  const T c = 0.5f * (vabs(t) - vabs(t - 2.0f * NB_BETA)) + NB_BETA;
  const T y = x - NB_BETA;
  const T r = 0.5f * (y + vabs(y));
  return NB_EPS * x + (1.0f - NB_EPS) * (c * c * (0.25f / NB_BETA) + r);
}

/* Input encoding: 12 values = wi, wo, half vector h, and the shortest-arc
 * difference vector  wd' = wi*hz + (v x wi) + (v.wi) v / (1 + hz),
 * v = (hy, -hx, 0)  (Rodrigues rotation of wi taking h to +Z). */
template<class T>
ccl_device_inline void encode12(const T wix, const T wiy, const T wiz,
                                const T wox, const T woy, const T woz,
                                T *ccl_restrict x)
{
  T hx = wix + wox, hy = wiy + woy, hz = wiz + woz;
  const T inv_len = T(1.0f) / vsqrt(kmax(hx * hx + hy * hy + hz * hz, 1e-20f));
  hx = hx * inv_len;
  hy = hy * inv_len;
  hz = hz * inv_len;

  const T vx = hy, vy = -hx; /* v = (hy, -hx, 0) */
  const T k = (vx * wix + vy * wiy) / (T(1.0f) + hz);
  /* v x wi with v.z == 0 */
  const T cx = vy * wiz;
  const T cy = -(vx * wiz);
  const T cz = vx * wiy - vy * wix;

  x[0] = wix;
  x[1] = wiy;
  x[2] = wiz;
  x[3] = wox;
  x[4] = woy;
  x[5] = woz;
  x[6] = hx;
  x[7] = hy;
  x[8] = hz;
  x[9] = wix * hz + cx + vx * k;
  x[10] = wiy * hz + cy + vy * k;
  x[11] = wiz * hz + cz;
}

constexpr int NB_ENC = 12;    /* directional encoding */
constexpr int NB_LATENT = 16; /* latent code */
constexpr int LANES = 8;      /* batch width (one AVX2 register of floats) */

/* Layer geometry. Layer l in [0, LAYERS]: l = 0 maps IN -> HID, 1..LAYERS-1 map
 * HID -> HID, l = LAYERS maps HID -> OUT. "LAYERS" counts hidden layers.      */
constexpr int mlp_layer_in(int l, int IN, int HID)
{
  return l == 0 ? IN : HID;
}
constexpr int mlp_layer_out(int l, int HID, int OUT, int LAYERS)
{
  return l == LAYERS ? OUT : HID;
}
constexpr int mlp_num_weights(int IN, int HID, int OUT, int LAYERS)
{
  int n = 0;
  for (int l = 0; l <= LAYERS; l++) {
    const int i = mlp_layer_in(l, IN, HID), o = mlp_layer_out(l, HID, OUT, LAYERS);
    n += i * o + o; /* weights + bias */
  }
  return n;
}
constexpr int mlp_num_macs(int IN, int HID, int OUT, int LAYERS)
{
  int n = 0;
  for (int l = 0; l <= LAYERS; l++)
    n += mlp_layer_in(l, IN, HID) * mlp_layer_out(l, HID, OUT, LAYERS);
  return n;
}

/* Weight count of the padded input-major layout used by dense_imv. */
constexpr int mlp_num_weights_padded(int IN, int HID, int OUT, int LAYERS)
{
  int n = 0;
  for (int l = 0; l <= LAYERS; l++) {
    const int i = mlp_layer_in(l, IN, HID), o = mlp_layer_out(l, HID, OUT, LAYERS);
    const int op = (o + 7) / 8 * 8;
    n += i * op + op;
  }
  return n;
}

/* Output-tile size of the batched kernels. mode 1 = one output neuron at a
 * time, mode 2 = register tiled (8/4/3 outputs at a time). */
constexpr int mlp_tile(int out, int mode)
{
  return mode == 1 ? 1 : (out % 8 == 0 ? 8 : (out % 4 == 0 ? 4 : (out % 3 == 0 ? 3 : 1)));
}

/* --- one eval, row-major weights: w[o*IN + i], bias[OUT] after the matrix -- */
template<int IN, int OUT, bool ACT>
ccl_device_inline void dense_rm(const float *ccl_restrict w,
                                const float *ccl_restrict in,
                                float *ccl_restrict out)
{
  const float *b = w + OUT * IN;
  for (int o = 0; o < OUT; o++) {
    const float *wr = w + o * IN;
    float acc = b[o];
    for (int i = 0; i < IN; i++) {
      acc += wr[i] * in[i];
    }
    out[o] = ACT ? leaky_smelu(acc) : acc;
  }
}

/* --- one eval, input-major weights: w[i*OUT + o], bias[OUT] after the matrix */
template<int IN, int OUT, bool ACT>
ccl_device_inline void dense_im(const float *ccl_restrict w,
                                const float *ccl_restrict in,
                                float *ccl_restrict out)
{
  const float *b = w + OUT * IN;
  float acc[OUT];
  for (int o = 0; o < OUT; o++) {
    acc[o] = b[o];
  }
  for (int i = 0; i < IN; i++) {
    const float xi = in[i];
    const float *wc = w + i * OUT;
    for (int o = 0; o < OUT; o++) {
      acc[o] += wc[o] * xi;
    }
  }
  for (int o = 0; o < OUT; o++) {
    out[o] = ACT ? leaky_smelu(acc[o]) : acc[o];
  }
}

#ifdef NB_HAVE_FLOAT8
/* --- one eval, input-major weights padded to a multiple of 8 outputs, explicit
 * 8-wide vectors over the OUTPUT dimension, SPLIT independent accumulator sets
 * over i (so SPLIT*V >= 8 FMA chains are in flight; a single chain is bound by
 * the 4-cycle FMA latency). Layout: w[i*OP + o] (OP = pad8(OUT)), bias[OP]
 * after the matrix. out receives OP floats.                                  */
constexpr int nb_pad8(int n)
{
  return (n + 7) / 8 * 8;
}
template<int IN, int OUT, bool ACT>
ccl_device_inline void dense_imv(const float *ccl_restrict w,
                                 const float *ccl_restrict in,
                                 float *ccl_restrict out)
{
  constexpr int V = nb_pad8(OUT) / 8;
  constexpr int OP = V * 8;
  constexpr int SPLIT = 8 / V;
  static_assert(IN % SPLIT == 0, "SPLIT must divide the input width");
  const float *b = w + IN * OP;
  float8 acc[SPLIT][V];
  for (int v = 0; v < V; v++) {
    acc[0][v] = load8(b + 8 * v);
    for (int s = 1; s < SPLIT; s++)
      acc[s][v] = float8(0.0f);
  }
  for (int i = 0; i < IN; i += SPLIT) {
    for (int s = 0; s < SPLIT; s++) {
      const float8 xi = float8(in[i + s]);
      const float *wc = w + (i + s) * OP;
      for (int v = 0; v < V; v++)
        acc[s][v] = madd(load8(wc + 8 * v), xi, acc[s][v]);
    }
  }
  for (int v = 0; v < V; v++) {
    float8 a = acc[0][v];
    for (int s = 1; s < SPLIT; s++)
      a = a + acc[s][v];
    store8(out + 8 * v, ACT ? leaky_smelu(a) : a);
  }
}
#endif

/* --- LANES evals batched, SoA activations in[i][lane], plain loops ---------
 * TILE output neurons are accumulated together so that one activation load
 * feeds TILE FMAs and TILE independent dependency chains hide FMA latency.   */
template<int IN, int OUT, int TILE, bool ACT>
ccl_device_inline void dense_b(const float *ccl_restrict w,
                               const float (*ccl_restrict in)[LANES],
                               float (*ccl_restrict out)[LANES])
{
  static_assert(OUT % TILE == 0, "tile must divide the layer width");
  const float *b = w + OUT * IN;
  for (int o0 = 0; o0 < OUT; o0 += TILE) {
    float acc[TILE][LANES];
    for (int t = 0; t < TILE; t++)
      for (int l = 0; l < LANES; l++)
        acc[t][l] = b[o0 + t];

    for (int i = 0; i < IN; i++) {
      for (int t = 0; t < TILE; t++) {
        const float wv = w[(o0 + t) * IN + i];
        for (int l = 0; l < LANES; l++)
          acc[t][l] += wv * in[i][l];
      }
    }

    for (int t = 0; t < TILE; t++)
      for (int l = 0; l < LANES; l++)
        out[o0 + t][l] = ACT ? leaky_smelu(acc[t][l]) : acc[t][l];
  }
}

#ifdef NB_HAVE_FLOAT8
/* --- LANES evals batched with an explicit 8-wide vector type ---------------- */
template<int IN, int OUT, int TILE, bool ACT>
ccl_device_inline void dense_v(const float *ccl_restrict w,
                               const float8 *ccl_restrict in,
                               float8 *ccl_restrict out)
{
  static_assert(OUT % TILE == 0, "tile must divide the layer width");
  const float *b = w + OUT * IN;
  for (int o0 = 0; o0 < OUT; o0 += TILE) {
    float8 acc[TILE];
    for (int t = 0; t < TILE; t++)
      acc[t] = float8(b[o0 + t]);

    for (int i = 0; i < IN; i++) {
      const float8 xi = in[i];
      for (int t = 0; t < TILE; t++)
        acc[t] = madd(float8(w[(o0 + t) * IN + i]), xi, acc[t]);
    }

    for (int t = 0; t < TILE; t++)
      out[o0 + t] = ACT ? leaky_smelu(acc[t]) : acc[t];
  }
}
#endif

template<int IN, int HID, int OUT, int LAYERS> struct NeuralMlp {
  static_assert(LAYERS >= 1, "need at least one hidden layer");
  static constexpr int num_weights = mlp_num_weights(IN, HID, OUT, LAYERS);

  /* One eval, row-major weights. x: IN floats. rgb: OUT floats (after exp). */
  ccl_device_inline void eval_rm(const float *ccl_restrict w,
                                 const float *ccl_restrict x,
                                 float *ccl_restrict rgb)
  {
    float buf0[HID], buf1[HID];
    dense_rm<IN, HID, true>(w, x, buf0);
    w += IN * HID + HID;
    float *cur = buf0, *nxt = buf1;
    for (int l = 1; l < LAYERS; l++) {
      dense_rm<HID, HID, true>(w, cur, nxt);
      w += HID * HID + HID;
      float *t = cur;
      cur = nxt;
      nxt = t;
    }
    dense_rm<HID, OUT, false>(w, cur, rgb);
    for (int c = 0; c < OUT; c++)
      rgb[c] = fast_expf(rgb[c]);
  }

  /* One eval, input-major weights. */
  ccl_device_inline void eval_im(const float *ccl_restrict w,
                                 const float *ccl_restrict x,
                                 float *ccl_restrict rgb)
  {
    float buf0[HID], buf1[HID];
    dense_im<IN, HID, true>(w, x, buf0);
    w += IN * HID + HID;
    float *cur = buf0, *nxt = buf1;
    for (int l = 1; l < LAYERS; l++) {
      dense_im<HID, HID, true>(w, cur, nxt);
      w += HID * HID + HID;
      float *t = cur;
      cur = nxt;
      nxt = t;
    }
    dense_im<HID, OUT, false>(w, cur, rgb);
    for (int c = 0; c < OUT; c++)
      rgb[c] = fast_expf(rgb[c]);
  }

#ifdef NB_HAVE_FLOAT8
  /* One eval, padded input-major weights, explicit float8 over outputs. */
  ccl_device_inline void eval_imv(const float *ccl_restrict w,
                                  const float *ccl_restrict x,
                                  float *ccl_restrict rgb)
  {
    constexpr int HP = nb_pad8(HID), OP = nb_pad8(OUT);
    float buf0[HP], buf1[HP], last[OP];
    dense_imv<IN, HID, true>(w, x, buf0);
    w += IN * HP + HP;
    float *cur = buf0, *nxt = buf1;
    for (int l = 1; l < LAYERS; l++) {
      dense_imv<HID, HID, true>(w, cur, nxt);
      w += HID * HP + HP;
      float *t = cur;
      cur = nxt;
      nxt = t;
    }
    dense_imv<HID, OUT, false>(w, cur, last);
    store8(last, fast_expf(load8(last))); /* OUT <= 8: one vector exp */
    for (int c = 0; c < OUT; c++)
      rgb[c] = last[c];
  }
#endif

  /* LANES evals at once, plain loops (SoA). x: [IN][LANES]. rgb: [OUT][LANES]. */
  template<int MODE>
  ccl_device_inline void eval_b(const float *ccl_restrict w,
                                const float (*ccl_restrict x)[LANES],
                                float (*ccl_restrict rgb)[LANES])
  {
    float buf0[HID][LANES], buf1[HID][LANES];
    dense_b<IN, HID, mlp_tile(HID, MODE), true>(w, x, buf0);
    w += IN * HID + HID;
    float (*cur)[LANES] = buf0;
    float (*nxt)[LANES] = buf1;
    for (int l = 1; l < LAYERS; l++) {
      dense_b<HID, HID, mlp_tile(HID, MODE), true>(w, cur, nxt);
      w += HID * HID + HID;
      float(*t)[LANES] = cur;
      cur = nxt;
      nxt = t;
    }
    dense_b<HID, OUT, mlp_tile(OUT, MODE), false>(w, cur, rgb);
    for (int c = 0; c < OUT; c++)
      for (int l = 0; l < LANES; l++)
        rgb[c][l] = fast_expf(rgb[c][l]);
  }

#ifdef NB_HAVE_FLOAT8
  /* LANES evals at once, explicit float8. x: [IN]. rgb: [OUT]. */
  template<int MODE>
  ccl_device_inline void eval_v(const float *ccl_restrict w,
                                const float8 *ccl_restrict x,
                                float8 *ccl_restrict rgb)
  {
    float8 buf0[HID], buf1[HID];
    dense_v<IN, HID, mlp_tile(HID, MODE), true>(w, x, buf0);
    w += IN * HID + HID;
    float8 *cur = buf0, *nxt = buf1;
    for (int l = 1; l < LAYERS; l++) {
      dense_v<HID, HID, mlp_tile(HID, MODE), true>(w, cur, nxt);
      w += HID * HID + HID;
      float8 *t = cur;
      cur = nxt;
      nxt = t;
    }
    dense_v<HID, OUT, mlp_tile(OUT, MODE), false>(w, cur, rgb);
    for (int c = 0; c < OUT; c++)
      rgb[c] = fast_expf(rgb[c]);
  }
#endif
};

/* ========================================================================= */
/* Benchmark harness (this part is NOT kernel style: STL, allocation, timing) */
/* ========================================================================= */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <functional>
#include <string>
#include <vector>

#ifndef BUILD_NAME
#  define BUILD_NAME "unnamed"
#endif

/* ---- RNG ---------------------------------------------------------------- */
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint32_t next_u32()
  { /* splitmix64 -> top bits */
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
  }
  float uniform() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }
  float range(float a, float b) { return a + (b - a) * uniform(); }
};

static void *xalloc(size_t bytes)
{
  void *p = nullptr;
  bytes = (bytes + 63) & ~size_t(63);
  if (posix_memalign(&p, 64, bytes) != 0 || !p) {
    fprintf(stderr, "alloc of %zu bytes failed\n", bytes);
    exit(1);
  }
  return p;
}

/* ---- Dataset: random directions / latents / analytic params -------------- */
/* AoS copy feeds the single-eval variants, SoA (blocks of LANES) copy feeds the
 * batched variants. Both hold exactly the same numbers.                       */
struct Dataset {
  size_t n = 0;
  float *dirs = nullptr;           /* [n][6]  wi.xyz wo.xyz            */
  float *lat = nullptr;            /* [n][16]                          */
  AnalyticParams *par = nullptr;   /* [n]                              */
  float *dirs_b = nullptr;         /* [n/LANES][6][LANES]              */
  float *lat_b = nullptr;          /* [n/LANES][16][LANES]             */
  float *par_b = nullptr;          /* [n/LANES][7][LANES]              */
};

static void cos_hemisphere(Rng &rng, float *d)
{
  const float u1 = rng.uniform(), u2 = rng.uniform();
  float z = fmaxf(sqrtf(1.0f - u1), 0.02f); /* keep away from grazing */
  const float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
  const float phi = 2.0f * M_PI_F * u2;
  d[0] = r * cosf(phi);
  d[1] = r * sinf(phi);
  d[2] = z;
}

static void make_dataset(Dataset &d, size_t n)
{
  d.n = n;
  d.dirs = (float *)xalloc(n * 6 * sizeof(float));
  d.lat = (float *)xalloc(n * NB_LATENT * sizeof(float));
  d.par = (AnalyticParams *)xalloc(n * sizeof(AnalyticParams));
  Rng rng(42);
  for (size_t i = 0; i < n; i++) {
    cos_hemisphere(rng, d.dirs + 6 * i);
    cos_hemisphere(rng, d.dirs + 6 * i + 3);
    for (int k = 0; k < NB_LATENT; k++)
      d.lat[i * NB_LATENT + k] = rng.range(-1.0f, 1.0f);
    AnalyticParams &p = d.par[i];
    p.rough = rng.range(0.05f, 1.0f);
    for (int c = 0; c < 3; c++) {
      p.f0[c] = rng.range(0.02f, 1.0f);
      p.albedo[c] = rng.range(0.0f, 1.0f);
    }
    p.pad = 0.0f;
  }
  const size_t nb = n / LANES;
  d.dirs_b = (float *)xalloc(nb * 6 * LANES * sizeof(float));
  d.lat_b = (float *)xalloc(nb * NB_LATENT * LANES * sizeof(float));
  d.par_b = (float *)xalloc(nb * 7 * LANES * sizeof(float));
  for (size_t b = 0; b < nb; b++) {
    for (int l = 0; l < LANES; l++) {
      const size_t i = b * LANES + l;
      for (int k = 0; k < 6; k++)
        d.dirs_b[(b * 6 + k) * LANES + l] = d.dirs[6 * i + k];
      for (int k = 0; k < NB_LATENT; k++)
        d.lat_b[(b * NB_LATENT + k) * LANES + l] = d.lat[i * NB_LATENT + k];
      const AnalyticParams &p = d.par[i];
      const float pv[7] = {p.rough, p.f0[0], p.f0[1], p.f0[2], p.albedo[0], p.albedo[1], p.albedo[2]};
      for (int k = 0; k < 7; k++)
        d.par_b[(b * 7 + k) * LANES + l] = pv[k];
    }
  }
}

/* ---- Weights ------------------------------------------------------------- */
/* Random He-initialised weights. Written in row-major layout (W[out][in] then
 * bias[out]) and, for the "im" variant, the exact same numbers transposed
 * (W[in][out] then bias[out]). */
template<int IN, int HID, int OUT, int LAYERS>
static void make_weights(float *rm, float *im, float *imp, uint64_t seed)
{
  Rng rng(seed);
  size_t off = 0, offp = 0;
  for (int k = 0; k < mlp_num_weights_padded(IN, HID, OUT, LAYERS); k++)
    imp[k] = 0.0f;
  for (int l = 0; l <= LAYERS; l++) {
    const int in = mlp_layer_in(l, IN, HID), out = mlp_layer_out(l, HID, OUT, LAYERS);
    const bool last = (l == LAYERS);
    const float a = last ? 0.5f * sqrtf(3.0f / in) : sqrtf(6.0f / in);
    for (int o = 0; o < out; o++)
      for (int i = 0; i < in; i++) {
        const float v = rng.range(-a, a);
        rm[off + (size_t)o * in + i] = v;
        im[off + (size_t)i * out + o] = v;
        imp[offp + (size_t)i * ((out + 7) / 8 * 8) + o] = v;
      }
    for (int o = 0; o < out; o++) {
      const float bv = last ? -0.7f : rng.range(-0.1f, 0.1f);
      rm[off + (size_t)out * in + o] = bv;
      im[off + (size_t)out * in + o] = bv;
      imp[offp + (size_t)in * ((out + 7) / 8 * 8) + o] = bv;
    }
    off += (size_t)in * out + out;
    offp += (size_t)in * ((out + 7) / 8 * 8) + (out + 7) / 8 * 8;
  }
}

/* ---- Runners: each evaluates [begin, begin+count) and returns a checksum --- */
static const size_t FLUSH = 4096; /* float partial sums are flushed to double every FLUSH evals */

template<bool S, bool D, bool P>
static double run_analytic(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  for (size_t n0 = begin; n0 < begin + count; n0 += FLUSH) {
    const size_t n1 = n0 + FLUSH < begin + count ? n0 + FLUSH : begin + count;
    float sr = 0, sg = 0, sb = 0, sp = 0;
    for (size_t n = n0; n < n1; n++) {
      const float *dir = d.dirs + 6 * n;
      const AnalyticParams &p = d.par[n];
      float rgb[3], pdf = 0.0f;
      analytic_eval<float, S, D, P>(dir[0], dir[1], dir[2], dir[3], dir[4], dir[5],
                                    p.rough, p.f0[0], p.f0[1], p.f0[2],
                                    p.albedo[0], p.albedo[1], p.albedo[2], rgb, &pdf);
      float r = rgb[0], g = rgb[1], b = rgb[2];
      BENCH_CONSUME(r);
      BENCH_CONSUME(g);
      BENCH_CONSUME(b);
      if (P)
        BENCH_CONSUME(pdf);
      sr += r;
      sg += g;
      sb += b;
      sp += pdf;
    }
    total += (double)sr + sg + sb + sp;
  }
  return total;
}

template<bool S, bool D, bool P>
static double run_analytic_b(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  const size_t b0 = begin / LANES, b1 = (begin + count) / LANES;
  for (size_t bb = b0; bb < b1; bb += FLUSH / LANES) {
    const size_t be = bb + FLUSH / LANES < b1 ? bb + FLUSH / LANES : b1;
    float s[4][LANES] = {};
    for (size_t b = bb; b < be; b++) {
      const float *dir = d.dirs_b + b * 6 * LANES;
      const float *par = d.par_b + b * 7 * LANES;
      float rgb[3][LANES], pdf[LANES];
      for (int l = 0; l < LANES; l++) {
        float o3[3], pp = 0.0f;
        analytic_eval<float, S, D, P>(dir[0 * LANES + l], dir[1 * LANES + l], dir[2 * LANES + l],
                                      dir[3 * LANES + l], dir[4 * LANES + l], dir[5 * LANES + l],
                                      par[0 * LANES + l], par[1 * LANES + l], par[2 * LANES + l],
                                      par[3 * LANES + l], par[4 * LANES + l], par[5 * LANES + l],
                                      par[6 * LANES + l], o3, &pp);
        rgb[0][l] = o3[0];
        rgb[1][l] = o3[1];
        rgb[2][l] = o3[2];
        pdf[l] = pp;
      }
      for (int l = 0; l < LANES; l++) {
        s[0][l] += rgb[0][l];
        s[1][l] += rgb[1][l];
        s[2][l] += rgb[2][l];
        s[3][l] += pdf[l];
      }
    }
    for (int c = 0; c < 4; c++)
      for (int l = 0; l < LANES; l++)
        total += s[c][l];
  }
  return total;
}

#ifdef NB_HAVE_FLOAT8
template<bool S, bool D, bool P>
static double run_analytic_v(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  const size_t b0 = begin / LANES, b1 = (begin + count) / LANES;
  for (size_t bb = b0; bb < b1; bb += FLUSH / LANES) {
    const size_t be = bb + FLUSH / LANES < b1 ? bb + FLUSH / LANES : b1;
    float8 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (size_t b = bb; b < be; b++) {
      const float *dir = d.dirs_b + b * 6 * LANES;
      const float *par = d.par_b + b * 7 * LANES;
      float8 rgb[3], pdf = 0.0f;
      analytic_eval<float8, S, D, P>(load8(dir + 0 * LANES), load8(dir + 1 * LANES), load8(dir + 2 * LANES),
                                     load8(dir + 3 * LANES), load8(dir + 4 * LANES), load8(dir + 5 * LANES),
                                     load8(par + 0 * LANES), load8(par + 1 * LANES), load8(par + 2 * LANES),
                                     load8(par + 3 * LANES), load8(par + 4 * LANES), load8(par + 5 * LANES),
                                     load8(par + 6 * LANES), rgb, &pdf);
      s0 = s0 + rgb[0];
      s1 = s1 + rgb[1];
      s2 = s2 + rgb[2];
      s3 = s3 + pdf;
    }
    float tmp[4][LANES];
    store8(tmp[0], s0);
    store8(tmp[1], s1);
    store8(tmp[2], s2);
    store8(tmp[3], s3);
    for (int c = 0; c < 4; c++)
      for (int l = 0; l < LANES; l++)
        total += tmp[c][l];
  }
  return total;
}
#endif

/* One MLP evaluation for sample n (single) / block b (batched). */
template<int IN, int HID, int LAYERS, int IM>
ccl_device_inline void mlp_eval_one(const Dataset &d, const float *w, size_t n, float *rgb)
{
  using Mlp = NeuralMlp<IN, HID, 3, LAYERS>;
  float x[IN];
  const float *dir = d.dirs + 6 * n;
  encode12<float>(dir[0], dir[1], dir[2], dir[3], dir[4], dir[5], x);
  if constexpr (IN > NB_ENC) {
    const float *lat = d.lat + NB_LATENT * n;
    for (int k = 0; k < NB_LATENT; k++)
      x[NB_ENC + k] = lat[k];
  }
#ifdef NB_HAVE_FLOAT8
  if constexpr (IM == 2)
    Mlp().eval_imv(w, x, rgb);
  else
#endif
  if constexpr (IM == 1)
    Mlp().eval_im(w, x, rgb);
  else
    Mlp().eval_rm(w, x, rgb);
}

template<int IN, int HID, int LAYERS, int MODE>
ccl_device_inline void mlp_eval_block(const Dataset &d, const float *w, size_t b, float (*rgb)[LANES])
{
  using Mlp = NeuralMlp<IN, HID, 3, LAYERS>;
  alignas(32) float x[IN][LANES];
  const float *dir = d.dirs_b + b * 6 * LANES;
  for (int l = 0; l < LANES; l++) {
    float xs[NB_ENC];
    encode12<float>(dir[0 * LANES + l], dir[1 * LANES + l], dir[2 * LANES + l],
                    dir[3 * LANES + l], dir[4 * LANES + l], dir[5 * LANES + l], xs);
    for (int k = 0; k < NB_ENC; k++)
      x[k][l] = xs[k];
  }
  if constexpr (IN > NB_ENC) {
    const float *lat = d.lat_b + b * NB_LATENT * LANES;
    for (int k = 0; k < NB_LATENT; k++)
      for (int l = 0; l < LANES; l++)
        x[NB_ENC + k][l] = lat[k * LANES + l];
  }
  Mlp().template eval_b<MODE>(w, x, rgb);
}

template<int IN, int HID, int LAYERS, int IM>
static double run_mlp_single(const Dataset &d, const float *w, size_t begin, size_t count)
{
  double total = 0.0;
  for (size_t n0 = begin; n0 < begin + count; n0 += FLUSH) {
    const size_t n1 = n0 + FLUSH < begin + count ? n0 + FLUSH : begin + count;
    float sr = 0, sg = 0, sb = 0;
    for (size_t n = n0; n < n1; n++) {
      float rgb[3];
      mlp_eval_one<IN, HID, LAYERS, IM>(d, w, n, rgb);
      float r = rgb[0], g = rgb[1], b = rgb[2];
      BENCH_CONSUME(r);
      BENCH_CONSUME(g);
      BENCH_CONSUME(b);
      sr += r;
      sg += g;
      sb += b;
    }
    total += (double)sr + sg + sb;
  }
  return total;
}

template<int IN, int HID, int LAYERS, int MODE>
static double run_mlp_batched(const Dataset &d, const float *w, size_t begin, size_t count)
{
  double total = 0.0;
  const size_t b0 = begin / LANES, b1 = (begin + count) / LANES;
  for (size_t bb = b0; bb < b1; bb += FLUSH / LANES) {
    const size_t be = bb + FLUSH / LANES < b1 ? bb + FLUSH / LANES : b1;
    float s[3][LANES] = {};
    for (size_t b = bb; b < be; b++) {
      float rgb[3][LANES];
      mlp_eval_block<IN, HID, LAYERS, MODE>(d, w, b, rgb);
      for (int c = 0; c < 3; c++)
        for (int l = 0; l < LANES; l++)
          s[c][l] += rgb[c][l];
    }
    for (int c = 0; c < 3; c++)
      for (int l = 0; l < LANES; l++)
        total += s[c][l];
  }
  return total;
}

#ifdef NB_HAVE_FLOAT8
template<int IN, int HID, int LAYERS, int MODE>
ccl_device_inline void mlp_eval_vec(const Dataset &d, const float *w, size_t b, float8 *rgb)
{
  using Mlp = NeuralMlp<IN, HID, 3, LAYERS>;
  float8 x[IN];
  const float *dir = d.dirs_b + b * 6 * LANES;
  encode12<float8>(load8(dir + 0 * LANES), load8(dir + 1 * LANES), load8(dir + 2 * LANES),
                   load8(dir + 3 * LANES), load8(dir + 4 * LANES), load8(dir + 5 * LANES), x);
  if constexpr (IN > NB_ENC) {
    const float *lat = d.lat_b + b * NB_LATENT * LANES;
    for (int k = 0; k < NB_LATENT; k++)
      x[NB_ENC + k] = load8(lat + k * LANES);
  }
  Mlp().template eval_v<MODE>(w, x, rgb);
}

template<int IN, int HID, int LAYERS, int MODE>
static double run_mlp_vec(const Dataset &d, const float *w, size_t begin, size_t count)
{
  double total = 0.0;
  const size_t b0 = begin / LANES, b1 = (begin + count) / LANES;
  for (size_t bb = b0; bb < b1; bb += FLUSH / LANES) {
    const size_t be = bb + FLUSH / LANES < b1 ? bb + FLUSH / LANES : b1;
    float8 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (size_t b = bb; b < be; b++) {
      float8 rgb[3];
      mlp_eval_vec<IN, HID, LAYERS, MODE>(d, w, b, rgb);
      s0 = s0 + rgb[0];
      s1 = s1 + rgb[1];
      s2 = s2 + rgb[2];
    }
    float tmp[3][LANES];
    store8(tmp[0], s0);
    store8(tmp[1], s1);
    store8(tmp[2], s2);
    for (int c = 0; c < 3; c++)
      for (int l = 0; l < LANES; l++)
        total += tmp[c][l];
  }
  return total;
}
#endif

/* ---- Micro-benchmarks that isolate parts of the decoder ------------------- */
static double run_encode_only(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  for (size_t n0 = begin; n0 < begin + count; n0 += FLUSH) {
    const size_t n1 = n0 + FLUSH < begin + count ? n0 + FLUSH : begin + count;
    float s = 0;
    for (size_t n = n0; n < n1; n++) {
      const float *dir = d.dirs + 6 * n;
      float x[NB_ENC];
      encode12<float>(dir[0], dir[1], dir[2], dir[3], dir[4], dir[5], x);
      float a = x[9], b = x[10], c = x[11];
      BENCH_CONSUME(a);
      BENCH_CONSUME(b);
      BENCH_CONSUME(c);
      s += a + b + c;
    }
    total += s;
  }
  return total;
}

template<bool FAST>
static double run_exp3(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  for (size_t n0 = begin; n0 < begin + count; n0 += FLUSH) {
    const size_t n1 = n0 + FLUSH < begin + count ? n0 + FLUSH : begin + count;
    float s = 0;
    for (size_t n = n0; n < n1; n++) {
      const float *lat = d.lat + NB_LATENT * n;
      float a = lat[0] * 4.0f, b = lat[1] * 4.0f, c = lat[2] * 4.0f;
      BENCH_CONSUME(a);
      BENCH_CONSUME(b);
      BENCH_CONSUME(c);
      float ea, eb, ec;
      if (FAST) {
        ea = fast_expf(a);
        eb = fast_expf(b);
        ec = fast_expf(c);
      }
      else {
        ea = expf(a);
        eb = expf(b);
        ec = expf(c);
      }
      BENCH_CONSUME(ea);
      BENCH_CONSUME(eb);
      BENCH_CONSUME(ec);
      s += ea + eb + ec;
    }
    total += s;
  }
  return total;
}

/* Memory-only baseline: reads everything a single MLP eval reads, does ~no math. */
static double run_stream(const Dataset &d, size_t begin, size_t count)
{
  double total = 0.0;
  for (size_t n0 = begin; n0 < begin + count; n0 += FLUSH) {
    const size_t n1 = n0 + FLUSH < begin + count ? n0 + FLUSH : begin + count;
    float s = 0;
    for (size_t n = n0; n < n1; n++) {
      const float *dir = d.dirs + 6 * n;
      const float *lat = d.lat + NB_LATENT * n;
      float a = dir[0] + dir[5], b = lat[0] + lat[8] + lat[15] + d.par[n].rough;
      BENCH_CONSUME(a);
      BENCH_CONSUME(b);
      s += a + b;
    }
    total += s;
  }
  return total;
}

/* ---- Peak throughput / clock estimation ----------------------------------- */
typedef float v8f __attribute__((vector_size(32)));

static double now_s()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double peak_vec_gflops()
{
  const long iters = 200000000;
  v8f a[12];
  for (int k = 0; k < 12; k++)
    for (int j = 0; j < 8; j++)
      a[k][j] = 1.0f + 0.01f * (k + j);
  const v8f b = {0.999f, 0.999f, 0.999f, 0.999f, 0.999f, 0.999f, 0.999f, 0.999f};
  const v8f c = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
  double best = 0;
  for (int rep = 0; rep < 3; rep++) {
    const double t0 = now_s();
    for (long it = 0; it < iters / 8; it++)
      for (int k = 0; k < 12; k++)
        a[k] = a[k] * b + c;
    const double t = now_s() - t0;
    best = fmax(best, (double)(iters / 8) * 12 * 8 * 2 / t * 1e-9);
  }
  float sink = 0;
  for (int k = 0; k < 12; k++)
    sink += a[k][0];
  if (sink == 12345.678f)
    printf("!");
  return best;
}

static double peak_scalar_gflops(double *ghz_est)
{
  const long iters = 200000000;
  const float b = 0.999f, c = 0.001f;
  double best = 0;
  for (int rep = 0; rep < 3; rep++) {
    float a[8];
    for (int k = 0; k < 8; k++)
      a[k] = 1.0f + 0.1f * k;
    const double t0 = now_s();
    for (long it = 0; it < iters / 8; it++)
      for (int k = 0; k < 8; k++) {
        a[k] = a[k] * b + c;
        BENCH_CONSUME(a[k]);
      }
    const double t = now_s() - t0;
    best = fmax(best, (double)(iters / 8) * 8 * 2 / t * 1e-9);
    float sink = 0;
    for (int k = 0; k < 8; k++)
      sink += a[k];
    if (sink == 12345.678f)
      printf("!");
  }
  /* Dependent FMA chain: one FMA per FMA-latency (assumed 4 cycles). */
  double bestt = 1e9;
  for (int rep = 0; rep < 3; rep++) {
    float a = 1.0f;
    const long n = 200000000;
    const double t0 = now_s();
    for (long it = 0; it < n; it++) {
      a = a * b + c;
      BENCH_CONSUME(a);
    }
    bestt = fmin(bestt, now_s() - t0);
    if (a == 12345.678f)
      printf("!");
  }
  *ghz_est = 4.0 * 200000000 / bestt * 1e-9;
  return best;
}

/* ---- Registry ------------------------------------------------------------ */
struct Variant {
  std::string name;
  std::function<double(size_t, size_t)> fn;
  double flops_mm = 0;  /* 2 * MACs */
  double flops_tot = 0; /* + activation + encoding + exp estimates */
};

struct Registry {
  std::vector<Variant> v;
  void add(const std::string &name, std::function<double(size_t, size_t)> fn, double fmm = 0, double ftot = 0)
  {
    v.push_back({name, fn, fmm, ftot});
  }
};

/* FLOP model for the decoder beyond the matmuls (documented in README):
 * LeakySmeLU ~ 11 flops per hidden neuron, encoding ~ 40, 3 x fast_expf ~ 15 each. */
constexpr double ACT_FLOPS = 11, ENC_FLOPS = 40, EXP_FLOPS = 15;

template<int IN, int HID, int LAYERS>
static void register_mlp(Registry &R, const Dataset &d, bool with_batched)
{
  using Mlp = NeuralMlp<IN, HID, 3, LAYERS>;
  float *wrm = (float *)xalloc(Mlp::num_weights * sizeof(float));
  float *wim = (float *)xalloc(Mlp::num_weights * sizeof(float));
  float *wimp = (float *)xalloc(mlp_num_weights_padded(IN, HID, 3, LAYERS) * sizeof(float));
  make_weights<IN, HID, 3, LAYERS>(wrm, wim, wimp, 1000 + IN * 7 + HID * 13 + LAYERS);
  const double fmm = 2.0 * mlp_num_macs(IN, HID, 3, LAYERS);
  const double ftot = fmm + ACT_FLOPS * HID * LAYERS + ENC_FLOPS + 3 * EXP_FLOPS;
  const std::string base = "mlp_h" + std::to_string(HID) + "x" + std::to_string(LAYERS) + "_in" + std::to_string(IN);
  R.add(base + "_rm", [=, &d](size_t b, size_t c) { return run_mlp_single<IN, HID, LAYERS, 0>(d, wrm, b, c); }, fmm, ftot);
  R.add(base + "_im", [=, &d](size_t b, size_t c) { return run_mlp_single<IN, HID, LAYERS, 1>(d, wim, b, c); }, fmm, ftot);
#ifdef NB_HAVE_FLOAT8
  R.add(base + "_imv", [=, &d](size_t b, size_t c) { return run_mlp_single<IN, HID, LAYERS, 2>(d, wimp, b, c); }, fmm, ftot);
#endif
  if (with_batched) {
    R.add(base + "_b8n", [=, &d](size_t b, size_t c) { return run_mlp_batched<IN, HID, LAYERS, 1>(d, wrm, b, c); }, fmm, ftot);
    R.add(base + "_b8t", [=, &d](size_t b, size_t c) { return run_mlp_batched<IN, HID, LAYERS, 2>(d, wrm, b, c); }, fmm, ftot);
#ifdef NB_HAVE_FLOAT8
    R.add(base + "_b8v", [=, &d](size_t b, size_t c) { return run_mlp_vec<IN, HID, LAYERS, 2>(d, wrm, b, c); }, fmm, ftot);
#endif
  }
}

/* ---- Verification --------------------------------------------------------- */
static double ref_smelu(double x)
{
  const double beta = NB_BETA, eps = NB_EPS;
  if (x < -beta)
    return eps * x;
  if (x > beta)
    return x;
  const double a = (1 - eps) / (4 * beta), b = (1 + eps) / 2, c = beta * (1 - eps) / 4;
  return a * x * x + b * x + c;
}

template<int IN, int HID, int LAYERS>
static void ref_mlp_double(const float *w, const float *x, double *rgb)
{
  double cur[64], nxt[64];
  for (int i = 0; i < IN; i++)
    cur[i] = x[i];
  size_t off = 0;
  for (int l = 0; l <= LAYERS; l++) {
    const int in = mlp_layer_in(l, IN, HID), out = mlp_layer_out(l, HID, 3, LAYERS);
    for (int o = 0; o < out; o++) {
      double acc = w[off + (size_t)out * in + o];
      for (int i = 0; i < in; i++)
        acc += (double)w[off + (size_t)o * in + i] * cur[i];
      nxt[o] = (l == LAYERS) ? acc : ref_smelu(acc);
    }
    for (int o = 0; o < out; o++)
      cur[o] = nxt[o];
    off += (size_t)in * out + out;
  }
  for (int c = 0; c < 3; c++)
    rgb[c] = exp(cur[c]);
}

static double rel(double a, double b)
{
  return fabs(a - b) / fmax(fabs(b), 1e-6);
}

template<int IN, int HID, int LAYERS>
static void verify_mlp(const Dataset &d)
{
  using Mlp = NeuralMlp<IN, HID, 3, LAYERS>;
  float *wrm = (float *)xalloc(Mlp::num_weights * sizeof(float));
  float *wim = (float *)xalloc(Mlp::num_weights * sizeof(float));
  float *wimp = (float *)xalloc(mlp_num_weights_padded(IN, HID, 3, LAYERS) * sizeof(float));
  make_weights<IN, HID, 3, LAYERS>(wrm, wim, wimp, 7);
  double e_rm = 0, e_im = 0, e_imv = 0, e_b1 = 0, e_b2 = 0, e_v = 0, mean = 0;
  const size_t nv = 2048;
  for (size_t b = 0; b < nv / LANES; b++) {
    float r1[3][LANES], r2[3][LANES];
    mlp_eval_block<IN, HID, LAYERS, 1>(d, wrm, b, r1);
    mlp_eval_block<IN, HID, LAYERS, 2>(d, wrm, b, r2);
#ifdef NB_HAVE_FLOAT8
    float8 rv8[3];
    float rv[3][LANES];
    mlp_eval_vec<IN, HID, LAYERS, 2>(d, wrm, b, rv8);
    for (int c = 0; c < 3; c++)
      store8(rv[c], rv8[c]);
#endif
    for (int l = 0; l < LANES; l++) {
      const size_t n = b * LANES + l;
      float x[IN], a[3], c[3];
      const float *dir = d.dirs + 6 * n;
      encode12<float>(dir[0], dir[1], dir[2], dir[3], dir[4], dir[5], x);
      if constexpr (IN > NB_ENC)
        for (int k = 0; k < NB_LATENT; k++)
          x[NB_ENC + k] = d.lat[NB_LATENT * n + k];
      double ref[3];
      ref_mlp_double<IN, HID, LAYERS>(wrm, x, ref);
      mlp_eval_one<IN, HID, LAYERS, 0>(d, wrm, n, a);
      mlp_eval_one<IN, HID, LAYERS, 1>(d, wim, n, c);
      float cx[3] = {0, 0, 0};
#ifdef NB_HAVE_FLOAT8
      mlp_eval_one<IN, HID, LAYERS, 2>(d, wimp, n, cx);
#endif
      for (int k = 0; k < 3; k++) {
        e_rm = fmax(e_rm, rel(a[k], ref[k]));
        e_im = fmax(e_im, rel(c[k], ref[k]));
        e_imv = fmax(e_imv, rel(cx[k], ref[k]));
        e_b1 = fmax(e_b1, rel(r1[k][l], ref[k]));
        e_b2 = fmax(e_b2, rel(r2[k][l], ref[k]));
#ifdef NB_HAVE_FLOAT8
        e_v = fmax(e_v, rel(rv[k][l], ref[k]));
#endif
        mean += ref[k] / (3.0 * nv);
      }
    }
  }
  printf("  mlp h%dx%d in%d: max rel err vs double ref: rm %.2e  im %.2e  imv %.2e  b8n %.2e  b8t %.2e  b8v %.2e  (mean output %.3f)\n",
         HID, LAYERS, IN, e_rm, e_im, e_imv, e_b1, e_b2, e_v, mean);
}

static int run_verify()
{
  printf("== verify (build %s) ==\n", BUILD_NAME);
  /* LeakySmeLU vs piecewise definition, and C1 continuity. */
  double maxe = 0;
  for (int i = -2000; i <= 2000; i++) {
    const float x = i * 0.001f;
    maxe = fmax(maxe, fabs(leaky_smelu(x) - ref_smelu(x)));
  }
  const double h = 1e-4;
  double dm = (ref_smelu(-NB_BETA) - ref_smelu(-NB_BETA - h)) / h, dp = (ref_smelu(-NB_BETA + h) - ref_smelu(-NB_BETA)) / h;
  double dm2 = (ref_smelu(NB_BETA) - ref_smelu(NB_BETA - h)) / h, dp2 = (ref_smelu(NB_BETA + h) - ref_smelu(NB_BETA)) / h;
  printf("  LeakySmeLU: max abs err branch-free vs piecewise %.2e; slope at -beta %.4f|%.4f, at +beta %.4f|%.4f (C1)\n",
         maxe, dm, dp, dm2, dp2);

  /* Encoding: h unit, wd' is the rotation of wi taking h to +Z => wd.z = wi.h, |wd| = 1. */
  Rng rng(5);
  double e_unit = 0, e_z = 0;
  for (int i = 0; i < 100000; i++) {
    float a[3], b[3], x[NB_ENC];
    cos_hemisphere(rng, a);
    cos_hemisphere(rng, b);
    encode12<float>(a[0], a[1], a[2], b[0], b[1], b[2], x);
    const double wd2 = (double)x[9] * x[9] + (double)x[10] * x[10] + (double)x[11] * x[11];
    const double wi_h = (double)a[0] * x[6] + (double)a[1] * x[7] + (double)a[2] * x[8];
    e_unit = fmax(e_unit, fabs(sqrt(wd2) - 1.0));
    e_z = fmax(e_z, fabs(x[11] - wi_h));
  }
  printf("  encode12: max ||wd|-1| %.2e, max |wd.z - wi.h| %.2e\n", e_unit, e_z);

  /* Analytic: directional albedo by Monte Carlo (uniform hemisphere over wo). */
  {
    const float th = 0.785398f;
    const float wix = sinf(th), wiy = 0.0f, wiz = cosf(th);
    for (float rough : {0.25f, 0.5f, 1.0f}) {
      double a_ggx = 0, a_lam = 0;
      const int ns = 2000000;
      Rng r2(11);
      for (int i = 0; i < ns; i++) {
        const float u1 = r2.uniform(), u2 = r2.uniform();
        const float z = u1, rr = sqrtf(fmaxf(1 - z * z, 0.0f)), phi = 2 * M_PI_F * u2;
        const float wox = rr * cosf(phi), woy = rr * sinf(phi), woz = z;
        float rgb[3], pdf;
        analytic_eval<float, true, false, false>(wix, wiy, wiz, wox, woy, woz, rough, 1.0f, 1.0f, 1.0f, 0, 0, 0, rgb, &pdf);
        a_ggx += rgb[0];
        analytic_eval<float, false, true, false>(wix, wiy, wiz, wox, woy, woz, rough, 0.0f, 0.0f, 0.0f, 1, 1, 1, rgb, &pdf);
        a_lam += rgb[0];
      }
      printf("  analytic albedo (wi=45deg, rough %.2f): GGX(F=1) %.3f (single-scatter: <=1, falls with roughness), Lambert(albedo 1, F0=0) %.3f (expect ~0.95 = 1 - coat Fresnel loss)\n",
             rough, 2 * M_PI * a_ggx / ns, 2 * M_PI * a_lam / ns);
    }
    /* scalar vs batched analytic must agree exactly (same source). */
  }
  {
    Dataset v;
    make_dataset(v, 4096);
    double s1 = run_analytic<true, true, true>(v, 0, 4096), s2 = run_analytic_b<true, true, true>(v, 0, 4096);
    double s3 = 0;
#ifdef NB_HAVE_FLOAT8
    s3 = run_analytic_v<true, true, true>(v, 0, 4096);
#endif
    printf("  analytic checksum scalar / lane-loop batched / float8 batched: %.6f / %.6f / %.6f\n", s1, s2, s3);
    verify_mlp<28, 16, 2>(v);
    verify_mlp<28, 32, 2>(v);
    verify_mlp<28, 64, 3>(v);
    verify_mlp<12, 16, 2>(v);
    verify_mlp<12, 32, 2>(v);
    verify_mlp<12, 64, 3>(v);
  }
  return 0;
}

/* ---- Main ----------------------------------------------------------------- */
static bool matches(const std::string &name, const std::vector<std::string> &only)
{
  if (only.empty())
    return true;
  for (const std::string &o : only)
    if (name.find(o) != std::string::npos)
      return true;
  return false;
}

static std::vector<std::string> split(const char *s)
{
  std::vector<std::string> out;
  std::string cur;
  for (; *s; s++) {
    if (*s == ',') {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
    }
    else
      cur += *s;
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

int main(int argc, char **argv)
{
  size_t N = 10000000;
  int reps = 3;
  size_t window = 2048;      /* cache-resident working set (evals) */
  size_t N64 = 0;            /* eval count for the h64x3 variants (0 = same as N) */
  double cache_thresh = 250; /* run the cache-resident pass if stream ns/eval is below this */
  std::vector<std::string> only, skip;
  bool verify = false, peak = true, in12 = true;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--verify")
      verify = true;
    else if (a == "--n" && i + 1 < argc)
      N = (size_t)atof(argv[++i]);
    else if (a == "--n64" && i + 1 < argc)
      N64 = (size_t)atof(argv[++i]);
    else if (a == "--reps" && i + 1 < argc)
      reps = atoi(argv[++i]);
    else if (a == "--only" && i + 1 < argc)
      only = split(argv[++i]);
    else if (a == "--skip" && i + 1 < argc)
      skip = split(argv[++i]);
    else if (a == "--cache-thresh" && i + 1 < argc)
      cache_thresh = atof(argv[++i]);
    else if (a == "--no-peak")
      peak = false;
    else if (a == "--no-in12")
      in12 = false;
    else if (a == "--list") {
    }
    else {
      fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 1;
    }
  }
  if (verify)
    return run_verify();

  N = (N / FLUSH) * FLUSH; /* multiple of 4096 (and of LANES) */
  if (N64 == 0 || N64 > N)
    N64 = N;
  N64 = (N64 / FLUSH) * FLUSH;
  printf("# build=%s compiler=%s N=%zu N(h64x3)=%zu reps=%d\n", BUILD_NAME, __VERSION__, N, N64, reps);
#ifdef BUILD_FLAGS
  printf("# flags=%s\n", BUILD_FLAGS);
#endif
  if (peak) {
    double ghz;
    const double ps = peak_scalar_gflops(&ghz);
    const double pv = peak_vec_gflops();
    printf("PEAK,%s,vec8_fma_gflops,%.1f,scalar_fma_gflops,%.1f,est_ghz,%.2f\n", BUILD_NAME, pv, ps, ghz);
    fflush(stdout);
  }

  Dataset d;
  make_dataset(d, N);

  Registry R;
  R.add("stream_baseline", [&](size_t b, size_t c) { return run_stream(d, b, c); });
  R.add("an_ggx", [&](size_t b, size_t c) { return run_analytic<true, false, false>(d, b, c); });
  R.add("an_lambert", [&](size_t b, size_t c) { return run_analytic<false, true, false>(d, b, c); });
  R.add("an_ggx_lambert", [&](size_t b, size_t c) { return run_analytic<true, true, false>(d, b, c); });
  R.add("an_ggx_lambert_pdf", [&](size_t b, size_t c) { return run_analytic<true, true, true>(d, b, c); });
  R.add("an_ggx_lambert_b8", [&](size_t b, size_t c) { return run_analytic_b<true, true, false>(d, b, c); });
  R.add("an_ggx_lambert_pdf_b8", [&](size_t b, size_t c) { return run_analytic_b<true, true, true>(d, b, c); });
#ifdef NB_HAVE_FLOAT8
  R.add("an_ggx_lambert_b8v", [&](size_t b, size_t c) { return run_analytic_v<true, true, false>(d, b, c); });
  R.add("an_ggx_lambert_pdf_b8v", [&](size_t b, size_t c) { return run_analytic_v<true, true, true>(d, b, c); });
#endif
  R.add("part_encode12", [&](size_t b, size_t c) { return run_encode_only(d, b, c); });
  R.add("part_exp3_fast", [&](size_t b, size_t c) { return run_exp3<true>(d, b, c); });
  R.add("part_exp3_libm", [&](size_t b, size_t c) { return run_exp3<false>(d, b, c); });
  register_mlp<28, 16, 2>(R, d, true);
  register_mlp<28, 32, 2>(R, d, true);
  register_mlp<28, 64, 3>(R, d, true);
  if (in12) {
    register_mlp<12, 16, 2>(R, d, true);
    register_mlp<12, 32, 2>(R, d, true);
    register_mlp<12, 64, 3>(R, d, true);
  }

  for (const Variant &v : R.v) {
    if (!matches(v.name, only) || (!skip.empty() && matches(v.name, skip)))
      continue;
    const size_t NV = (v.name.find("h64x3") != std::string::npos) ? N64 : N;
    double best = 1e30, worst = 0, cs = 0;
    for (int r = 0; r < reps; r++) {
      const double t0 = now_s();
      cs = v.fn(0, NV);
      const double t = (now_s() - t0) / NV * 1e9;
      best = fmin(best, t);
      worst = fmax(worst, t);
    }
    double best_c = -1;
    if (best < cache_thresh) {
      best_c = 1e30;
      for (int r = 0; r < reps; r++) {
        const double t0 = now_s();
        double c2 = 0;
        for (size_t k = 0; k < NV / window; k++)
          c2 += v.fn(0, window);
        best_c = fmin(best_c, (now_s() - t0) / ((NV / window) * window) * 1e9);
        (void)c2;
      }
    }
    printf("RESULT,%s,%s,%.3f,%.3f,%.3f,%.0f,%.0f,%.5f\n", BUILD_NAME, v.name.c_str(), best, worst, best_c,
           v.flops_mm, v.flops_tot, cs / N);
    fflush(stdout);
  }
  return 0;
}
