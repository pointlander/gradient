// Copyright 2022 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arm_neon.h>
#include <stdint.h>

void vdot(float *a, float *b, long n, float* ret) {
    int epoch = n / 4;
    int remain = n % 4;
    float32x4_t s;
    if (epoch > 0) {
        float32x4_t v1 = vld1q_f32(a);
        float32x4_t v2 = vld1q_f32(b);
        s = vmulq_f32(v1, v2);
        a += 4;
        b += 4;
    }
    for (int i = 1; i < epoch; i++) {
        float32x4_t v1 = vld1q_f32(a);
        float32x4_t v2 = vld1q_f32(b);
        s = vmlaq_f32(s, v1, v2);
        a += 4;
        b += 4;
    }
    float partial[4];
    vst1q_f32(partial, s);
    *ret = 0;
    for (int i = 0; i < 4; i++) {
        *ret += partial[i];
    }
    for (int i = 0; i < remain; i++) {
        *ret += a[i] * b[i];
    }
}