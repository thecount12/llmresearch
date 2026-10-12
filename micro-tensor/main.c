#include <stdio.h>
#include "tensor.h"

/* Four XOR rows: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0. */
static const float xor_in[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};
static const float xor_target[4] = {0.0f, 1.0f, 1.0f, 0.0f};

static int xor_forward(const Tensor *x,
                       const Tensor *W1, const Tensor *b1,
                       const Tensor *W2, const Tensor *b2,
                       Tensor *xw, Tensor *z1, Tensor *h,
                       Tensor *hw, Tensor *y) {
    if (!tensor_matmul(x, W1, xw) || !tensor_add_bias(xw, b1, z1) ||
        !tensor_relu(z1, h) || !tensor_matmul(h, W2, hw) ||
        !tensor_add_bias(hw, b2, y)) {
        return 0;
    }
    return 1;
}

/* One sample of h = relu(x @ W1 + b1), y = h @ W2 + b2.
 * Loss is 0.5 * (y - target)^2. Weights move by -lr * gradient.
 */
static int xor_step(Tensor *x, float target,
                    Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
                    Tensor *xw, Tensor *z1, Tensor *h, Tensor *hw, Tensor *y,
                    Tensor *dy, Tensor *dh, Tensor *dz1,
                    Tensor *dW1, Tensor *dW2,
                    float lr, float *loss) {
    int j;
    float err;

    if (!xor_forward(x, W1, b1, W2, b2, xw, z1, h, hw, y)) {
        return 0;
    }

    err = y->data[0] - target;
    *loss += 0.5f * err * err;
    dy->data[0] = err;

    /* dW2 = h^T * dy,  dh = dy * W2^T */
    if (!tensor_outer(h, dy, dW2)) {
        return 0;
    }
    for (j = 0; j < h->cols; j++) {
        dh->data[j] = err * W2->data[j];
    }
    if (!tensor_relu_grad(dh, z1, dz1) || !tensor_outer(x, dz1, dW1)) {
        return 0;
    }

    if (!tensor_scale_add(W1, dW1, -lr) ||
        !tensor_scale_add(b1, dz1, -lr) ||
        !tensor_scale_add(W2, dW2, -lr)) {
        return 0;
    }
    b2->data[0] -= lr * err;
    return 1;
}

static int xor_loss(Tensor *x,
                    Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
                    Tensor *xw, Tensor *z1, Tensor *h, Tensor *hw, Tensor *y,
                    float *loss) {
    int s;

    *loss = 0.0f;
    for (s = 0; s < 4; s++) {
        float err;
        x->data[0] = xor_in[s][0];
        x->data[1] = xor_in[s][1];
        if (!xor_forward(x, W1, b1, W2, b2, xw, z1, h, hw, y)) {
            return 0;
        }
        err = y->data[0] - xor_target[s];
        *loss += 0.5f * err * err;
    }
    return 1;
}

static void xor_print_preds(Tensor *x,
                            Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
                            Tensor *xw, Tensor *z1, Tensor *h,
                            Tensor *hw, Tensor *y) {
    int s;

    for (s = 0; s < 4; s++) {
        x->data[0] = xor_in[s][0];
        x->data[1] = xor_in[s][1];
        if (!xor_forward(x, W1, b1, W2, b2, xw, z1, h, hw, y)) {
            printf("XOR forward failed\n");
            return;
        }
        printf("  (%.0f, %.0f) -> %5.2f   target %.0f\n",
               xor_in[s][0], xor_in[s][1], y->data[0], xor_target[s]);
    }
}

/* Train the same two-layer net on XOR. Returns 0 when each
 * prediction lands within 0.15 of its target.
 */
static int train_xor(void) {
    const int epochs = 4000;
    const float lr = 0.1f;
    Tensor *x = tensor_alloc(1, 2);
    Tensor *W1 = tensor_alloc(2, 2);
    Tensor *b1 = tensor_alloc(1, 2);
    Tensor *W2 = tensor_alloc(2, 1);
    Tensor *b2 = tensor_alloc(1, 1);
    Tensor *xw = tensor_alloc(1, 2);
    Tensor *z1 = tensor_alloc(1, 2);
    Tensor *h = tensor_alloc(1, 2);
    Tensor *hw = tensor_alloc(1, 1);
    Tensor *y = tensor_alloc(1, 1);
    Tensor *dy = tensor_alloc(1, 1);
    Tensor *dh = tensor_alloc(1, 2);
    Tensor *dz1 = tensor_alloc(1, 2);
    Tensor *dW1 = tensor_alloc(2, 2);
    Tensor *dW2 = tensor_alloc(2, 1);
    float loss = 0.0f;
    int epoch;
    int s;
    int ok = 1;

    if (!x || !W1 || !b1 || !W2 || !b2 || !xw || !z1 || !h || !hw || !y ||
        !dy || !dh || !dz1 || !dW1 || !dW2) {
        printf("Memory allocation failed.\n");
        ok = 0;
    }

    if (ok) {
        /* Fixed start, not the XOR solution. */
        W1->data[0] = 0.6f;
        W1->data[1] = -0.4f;
        W1->data[2] = -0.3f;
        W1->data[3] = 0.8f;
        b1->data[0] = 0.1f;
        b1->data[1] = -0.2f;
        W2->data[0] = 0.5f;
        W2->data[1] = -0.4f;
        b2->data[0] = 0.1f;

        printf("XOR before training\n");
        tensor_print(W1, "W1");
        tensor_print(W2, "W2");
        if (!xor_loss(x, W1, b1, W2, b2, xw, z1, h, hw, y, &loss)) {
            ok = 0;
        } else {
            printf("loss %6.3f\n", loss);
            xor_print_preds(x, W1, b1, W2, b2, xw, z1, h, hw, y);
        }
    }

    for (epoch = 0; ok && epoch < epochs; epoch++) {
        loss = 0.0f;
        for (s = 0; s < 4; s++) {
            x->data[0] = xor_in[s][0];
            x->data[1] = xor_in[s][1];
            if (!xor_step(x, xor_target[s], W1, b1, W2, b2,
                          xw, z1, h, hw, y, dy, dh, dz1, dW1, dW2,
                          lr, &loss)) {
                printf("XOR step failed\n");
                ok = 0;
                break;
            }
        }
    }

    if (ok) {
        printf("XOR after %d epochs, lr %.2f\n", epochs, lr);
        tensor_print(W1, "W1");
        tensor_print(W2, "W2");
        if (!xor_loss(x, W1, b1, W2, b2, xw, z1, h, hw, y, &loss)) {
            ok = 0;
        } else {
            printf("loss %6.3f\n", loss);
            xor_print_preds(x, W1, b1, W2, b2, xw, z1, h, hw, y);
            for (s = 0; s < 4; s++) {
                float err;
                x->data[0] = xor_in[s][0];
                x->data[1] = xor_in[s][1];
                if (!xor_forward(x, W1, b1, W2, b2, xw, z1, h, hw, y)) {
                    ok = 0;
                    break;
                }
                err = y->data[0] - xor_target[s];
                if (err < 0.0f) {
                    err = -err;
                }
                if (err > 0.15f) {
                    printf("XOR prediction missed target by %.2f\n", err);
                    ok = 0;
                    break;
                }
            }
        }
    }

    tensor_free(x);
    tensor_free(W1);
    tensor_free(b1);
    tensor_free(W2);
    tensor_free(b2);
    tensor_free(xw);
    tensor_free(z1);
    tensor_free(h);
    tensor_free(hw);
    tensor_free(y);
    tensor_free(dy);
    tensor_free(dh);
    tensor_free(dz1);
    tensor_free(dW1);
    tensor_free(dW2);
    return ok;
}

int main(void) {
    /* Create a 2x3 and a 3x2 matrix */
    Tensor *A = tensor_alloc(2, 3);
    Tensor *B = tensor_alloc(3, 2);
    Tensor *C = tensor_alloc(2, 2);

    if (!A || !B || !C) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    /* Populate test data */
    tensor_fill(A, 2.0f); /* [[2, 2, 2], [2, 2, 2]] */
    tensor_fill(B, 3.0f); /* [[3, 3], [3, 3], [3, 3]] */

    /* Run multiplication */
    if (tensor_matmul(A, B, C)) {
        tensor_print(A, "A");
        tensor_print(B, "B");
        tensor_print(C, "Result C (A * B)");
    } else {
        printf("Dimension mismatch error!\n");
    }

    /* Cleanup memory safely */
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);

    /* y = relu(x @ W + b)
     * x = [1, 2], W = I, b = [0, -3]
     * x @ W + b = [1, -1], relu = [1, 0]
     */
    {
        Tensor *x = tensor_alloc(1, 2);
        Tensor *W = tensor_alloc(2, 2);
        Tensor *bias = tensor_alloc(1, 2);
        Tensor *xw = tensor_alloc(1, 2);
        Tensor *xwb = tensor_alloc(1, 2);
        Tensor *y = tensor_alloc(1, 2);

        if (!x || !W || !bias || !xw || !xwb || !y) {
            printf("Memory allocation failed.\n");
            tensor_free(x);
            tensor_free(W);
            tensor_free(bias);
            tensor_free(xw);
            tensor_free(xwb);
            tensor_free(y);
            return 1;
        }

        x->data[0] = 1.0f;
        x->data[1] = 2.0f;
        W->data[0] = 1.0f;
        W->data[1] = 0.0f;
        W->data[2] = 0.0f;
        W->data[3] = 1.0f;
        bias->data[0] = 0.0f;
        bias->data[1] = -3.0f;

        if (!tensor_matmul(x, W, xw) ||
            !tensor_add_bias(xw, bias, xwb) ||
            !tensor_relu(xwb, y)) {
            printf("Forward pass dimension mismatch!\n");
            tensor_free(x);
            tensor_free(W);
            tensor_free(bias);
            tensor_free(xw);
            tensor_free(xwb);
            tensor_free(y);
            return 1;
        }

        tensor_print(x, "x");
        tensor_print(W, "W");
        tensor_print(bias, "b");
        tensor_print(xwb, "x @ W + b");
        tensor_print(y, "relu(x @ W + b)");

        /* y2 = h @ W2 + b2, with h = relu(x @ W + b) = [1, 0]
         * W2 = [[2], [4]], b2 = [-1]
         * h @ W2 = 2, plus bias = 1
         */
        {
            Tensor *W2 = tensor_alloc(2, 1);
            Tensor *b2 = tensor_alloc(1, 1);
            Tensor *hw = tensor_alloc(1, 1);
            Tensor *y2 = tensor_alloc(1, 1);

            if (!W2 || !b2 || !hw || !y2) {
                printf("Memory allocation failed.\n");
                tensor_free(x);
                tensor_free(W);
                tensor_free(bias);
                tensor_free(xw);
                tensor_free(xwb);
                tensor_free(y);
                tensor_free(W2);
                tensor_free(b2);
                tensor_free(hw);
                tensor_free(y2);
                return 1;
            }

            W2->data[0] = 2.0f;
            W2->data[1] = 4.0f;
            b2->data[0] = -1.0f;

            if (!tensor_matmul(y, W2, hw) || !tensor_add_bias(hw, b2, y2)) {
                printf("Hidden layer dimension mismatch!\n");
                tensor_free(x);
                tensor_free(W);
                tensor_free(bias);
                tensor_free(xw);
                tensor_free(xwb);
                tensor_free(y);
                tensor_free(W2);
                tensor_free(b2);
                tensor_free(hw);
                tensor_free(y2);
                return 1;
            }

            tensor_print(W2, "W2");
            tensor_print(b2, "b2");
            tensor_print(y2, "h @ W2 + b2");

            tensor_free(W2);
            tensor_free(b2);
            tensor_free(hw);
            tensor_free(y2);
        }

        tensor_free(x);
        tensor_free(W);
        tensor_free(bias);
        tensor_free(xw);
        tensor_free(xwb);
        tensor_free(y);
    }

    if (!train_xor()) {
        return 1;
    }
    return 0;
}


