#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "basic_cnn_cifar100_brevitas_w4a4_params.h"
#include "images.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;
    if (argc > 1) {
        if (strcmp(argv[1], "cpu") == 0) tiled_matmul_type = CPU;
        else if (strcmp(argv[1], "os") == 0) tiled_matmul_type = OS;
        else if (strcmp(argv[1], "ws") == 0) tiled_matmul_type = WS;
    }

    uint64_t start, end;
    uint64_t conv_cycles = 0, matmul_cycles = 0;

    // model execution
    // conv_1
    start = read_cycles();
    tiled_conv_auto(conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim, conv_1_params.in_channels, conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim, conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size, false, false, false, false, false, (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out, NO_ACTIVATION, conv_1_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_2
    start = read_cycles();
    tiled_conv_auto(conv_2_params.batch_size, conv_2_params.in_row_dim, conv_2_params.in_col_dim, conv_2_params.in_channels, conv_2_params.out_channels, conv_2_params.out_row_dim, conv_2_params.out_col_dim, conv_2_params.stride, 1, 1, conv_2_params.padding, conv_2_params.kernel_size, false, false, false, false, false, (elem_t*)conv_1_out, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out, NO_ACTIVATION, conv_2_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_3
    start = read_cycles();
    tiled_conv_auto(conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim, conv_3_params.in_channels, conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim, conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size, false, false, false, false, false, (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out, NO_ACTIVATION, conv_3_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_4
    start = read_cycles();
    tiled_conv_auto(conv_4_params.batch_size, conv_4_params.in_row_dim, conv_4_params.in_col_dim, conv_4_params.in_channels, conv_4_params.out_channels, conv_4_params.out_row_dim, conv_4_params.out_col_dim, conv_4_params.stride, 1, 1, conv_4_params.padding, conv_4_params.kernel_size, false, false, false, false, false, (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out, NO_ACTIVATION, conv_4_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_5
    start = read_cycles();
    tiled_conv_auto(conv_5_params.batch_size, conv_5_params.in_row_dim, conv_5_params.in_col_dim, conv_5_params.in_channels, conv_5_params.out_channels, conv_5_params.out_row_dim, conv_5_params.out_col_dim, conv_5_params.stride, 1, 1, conv_5_params.padding, conv_5_params.kernel_size, false, false, false, false, false, (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out, NO_ACTIVATION, conv_5_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_6
    start = read_cycles();
    tiled_conv_auto(conv_6_params.batch_size, conv_6_params.in_row_dim, conv_6_params.in_col_dim, conv_6_params.in_channels, conv_6_params.out_channels, conv_6_params.out_row_dim, conv_6_params.out_col_dim, conv_6_params.stride, 1, 1, conv_6_params.padding, conv_6_params.kernel_size, false, false, false, false, false, (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out, NO_ACTIVATION, conv_6_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_7
    start = read_cycles();
    tiled_conv_auto(conv_7_params.batch_size, conv_7_params.in_row_dim, conv_7_params.in_col_dim, conv_7_params.in_channels, conv_7_params.out_channels, conv_7_params.out_row_dim, conv_7_params.out_col_dim, conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size, false, false, false, false, false, (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out, NO_ACTIVATION, conv_7_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // conv_8
    start = read_cycles();
    tiled_conv_auto(conv_8_params.batch_size, conv_8_params.in_row_dim, conv_8_params.in_col_dim, conv_8_params.in_channels, conv_8_params.out_channels, conv_8_params.out_row_dim, conv_8_params.out_col_dim, conv_8_params.stride, 1, 1, conv_8_params.padding, conv_8_params.kernel_size, false, false, false, false, false, (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out, NO_ACTIVATION, conv_8_params.output_scale, 1, 1, 0, tiled_matmul_type);
    end = read_cycles();
    conv_cycles += end - start;
    // fc_9
    start = read_cycles();
    tiled_matmul_nn_auto(4, 512, 2048, (elem_t*)conv_8_out, (elem_t*)fc_9_w, (acc_t*)fc_9_b, (elem_t*)fc_9_out, NO_ACTIVATION, 1.0, true, tiled_matmul_type, false, "fc_9");
    end = read_cycles();
    matmul_cycles += end - start;
    // fc_10
    start = read_cycles();
    tiled_matmul_nn_auto(4, 100, 512, (elem_t*)fc_9_out, (elem_t*)fc_10_w, (acc_t*)fc_10_b, (elem_t*)fc_10_out, NO_ACTIVATION, 1.0, true, tiled_matmul_type, false, "fc_10");
    end = read_cycles();
    matmul_cycles += end - start;

    uint64_t total_cycles = conv_cycles + matmul_cycles;
    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    if(total_cycles != 0) {
        printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (int)((matmul_cycles * 100) / total_cycles));
        printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (int)((conv_cycles * 100) / total_cycles));
    }
    printf("PASS\n");

    exit(0);
}
