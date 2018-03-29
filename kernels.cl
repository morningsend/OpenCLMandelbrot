
float2 complex_multiply(float2 a, float2 b) {
    return (float2) (a.x * a.x - b.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void mandelbrot(global int* image, float2 c, const int W, const int H) {

    float2 center = (float2) (0.0f, 0.0f);
    int i = get_global_id(0);
    int j = get_global_id(1);

    float2 complexCoord = (float2)((float) i / W, (float) j / H);
    complexCoord = (complexCoord * 4 - (float2)(2.0f, 2.0f) - c) * 0.75f;
    float2 z = complexCoord;
    int iter = 0;
    float mod2 = 0;
    while(iter < 20000 && mod2 < 4.0){
        z = complex_multiply(z, z) + complexCoord;
        mod2 = dot(z, z);
        iter++;
    }
    image[i + j * W] = iter;
}

kernel void matrix_multiply_v1 (
        const int M,
        const global float* A,
        const global float* B,
        global float* result
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int K = M;
    float accum = 0.0f;
    for(int k = 0; k < K; k++) {
        accum += A[k * M + i] * B[k*j + k];
    }

    result[i * M + j] = (float) 3;
}