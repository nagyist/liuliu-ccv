#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("convolutional network of 11x11 on 225x185 with uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 225, 185, 3), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 55, 45, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 11, 11, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 11, 11, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 11 * 11 * 3 * 4; i++)
		w->data.f32[i] = 1;
	for (i = 0; i < 225 * 185 * 3; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 45, CCV_32F | 4, 0, 0);
	int x, y;
	for (y = 0; y < 55; y++)
		for (x = 0; x < 45; x++)
			for (i = 0; i < 4; i++)
			c->data.f32[(y * 45 + x) * 4 + i] = ((x == 0 && y == 0) || (x == 0 && y == 54) || (x == 44 && y == 0) || (x == 44 && y == 54)) ? 300 : ((x == 0 || y == 0 || x == 44 || y == 54) ? 330 : 363);
	REQUIRE_MATRIX_EQ(b, c, "55x45 matrix should be exactly a matrix fill 363, with 300 on the corner and 330 on the border");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 5x3 on 17x27 with uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 27, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 27, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 5, 3, 1);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 5, 3, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 5 * 3 * 4; i++)
		w->data.f32[i] = 1;
	for (i = 0; i < 17 * 27; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(17, 27, CCV_32F | 4, 0, 0);
	int x, y;
	for (y = 0; y < 17; y++)
		for (x = 0; x < 27; x++)
			for (i = 0; i < 4; i++)
			{
				if ((x == 0 && y == 0) || (x == 0 && y == 16) || (x == 26 && y == 0) || (x == 26 && y == 16))
					c->data.f32[(y * 27 + x) * 4 + i] = 6;
				else if ((x == 0 && y == 1) || (x == 26 && y == 1) || (x == 0 && y == 15) || (x == 26 && y == 15))
					c->data.f32[(y * 27 + x) * 4 + i] = 8;
				else if (y == 0 || y == 16)
					c->data.f32[(y * 27 + x) * 4 + i] = 9;
				else if (x == 0 || x == 26)
					c->data.f32[(y * 27 + x) * 4 + i] = 10;
				else if (y == 1 || y == 15)
					c->data.f32[(y * 27 + x) * 4 + i] = 12;
				else
					c->data.f32[(y * 27 + x) * 4 + i] = 15;
			}
	REQUIRE_MATRIX_EQ(b, c, "17x27 matrix should be exactly a matrix fill 15, with 6, 8 on the corner and 9, 10, 12 on the border");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 11x11x11 on 33x225x185 with uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 33, 225, 185, 3), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 55, 45, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 11, 11, 11, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 4;
	hint.border.begin[0] = 1;
	hint.border.end[0] = 1;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 11, 11, 11, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 11 * 11 * 11 * 3 * 4; i++)
		w->data.f32[i] = 1;
	for (i = 0; i < 33 * 225 * 185 * 3; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 55, 45, 4), 0);
	int x, y, z;
	for (z = 0; z < 7; z++)
		for (y = 0; y < 55; y++)
			for (x = 0; x < 45; x++)
				for (i = 0; i < 4; i++)
					c->data.f32[((z * 55 + y) * 45 + x) * 4 + i] = ((x == 0 && y == 0 && z == 0) || (x == 0 && y == 0 && z == 6) || (x == 0 && y == 54 && z == 0) || (x == 0 && y == 54 && z == 6) || (x == 44 && y == 0 && z == 0) || (x == 44 && y == 0 && z == 6) || (x == 44 && y == 54 && z == 0) || (x == 44 && y == 54 && z == 6)) ? 3000 : (((x == 0 && y == 0) || (x == 0 && z == 0) || (y == 0 && z == 0) || (x == 0 && y == 54) || (x == 0 && z == 6) || (y == 0 && z == 6) || (x == 44 && y == 0) || (x == 44 && z == 0) || (y == 54 && z == 0) || (x == 44 && y == 54) || (x == 44 && z == 6) || (y == 54 && z == 6)) ? 3300 : ((x == 0 || x == 44 || y == 0 || y == 54 || z == 0 || z == 6) ? 3630 : 3993));
	REQUIRE_TENSOR_EQ(b, c, "7x55x45 matrix should be exactly a matrix fill 3993, with 3000 on the corner and 3300 on the ridge and 3630 on the surface");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 11x11 on 225x185 with non-uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 225, 185, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 55, 45, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 11, 11, 1);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 11, 11, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	// configure the inlets.
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 11 * 11; i++)
			w->data.f32[x * 11 * 11 + i] = i + 1;
	for (i = 0; i < 225 * 185; i++)
		a->data.f32[i] = i + 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 45, CCV_32F | 4, 0, 0);
	float sum = 0;
	// first column
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 2) * (y * 185 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[i] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 185 + (x + 3) + 1);
	for (x = 1; x < 44; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[x * 4 + i] = sum + (x - 1) * 4 * (11 * 11 + 12) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 185 + (x + 175) + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[44 * 4 + i] = sum;
	// last column
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 2) * ((y + 215) * 185 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[54 * 45 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 185 + (x + 3) + 1);
	for (x = 1; x < 44; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(54 * 45 + x) * 4 + i] = sum + (x - 1) * 4 * (10 * 11 + 1) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 185 + (x + 175) + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[(54 * 45 + 44) * 4 + i] = sum;
	float border[] = {
		0, 0
	};
	for (y = 0; y < 11; y++)
		for (x = 0; x < 10; x++)
			border[0] += (y * 11 + x + 2) * ((y + 3) * 185 + x + 1);
	for (y = 0; y < 11; y++)
		for (x = 0; x < 10; x++)
			border[1] += (y * 11 + x + 1) * ((y + 3) * 185 + (x + 175) + 1);
	sum = 0;
	for (y = 0; y < 11; y++)
		for (x = 0; x < 11; x++)
			sum += (y * 11 + x + 1) * ((y + 3) * 185 + (x + 3) + 1);
	for (y = 1; y < 54; y++)
	{
		for (i = 0; i < 4; i++)
			c->data.f32[y * 45 * 4 + i] = border[0];
		for (x = 1; x < 44; x++)
			for (i = 0; i < 4; i++)
				c->data.f32[(y * 45 + x) * 4 + i] = sum + (x - 1) * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		for (i = 0; i < 4; i++)
			c->data.f32[(y * 45 + 44) * 4 + i] = border[1];
		sum += 185 * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		border[0] += 185 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (10 * 11 + 1 + 1) * 11 / 2);
		border[1] += 185 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (11 * 11 + 11) * 11 / 2);
	}
	// regularize the output so it is within the tolerance
	for (i = 0; i < 55 * 45 * 4; i++)
		c->data.f32[i] = c->data.f32[i] * 1e-7, b->data.f32[i] = b->data.f32[i] * 1e-7;
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 3x5 on 27x27 with non-uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 3, 5, 1);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3, 5, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	// configure the inlets.
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 3 * 5; i++)
			w->data.f32[x * 3 * 5 + i] = i + 1;
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = i + 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | 4, 0, 0);
	// the first column
	float sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 3) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 2) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[4 + i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 5; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[x * 4 + i] = sum + (x - 2) * 21 * 10 / 2;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[25 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[26 * 4 + i] = sum;
	// the last column
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 3) * ((y + 25) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[27 * 26 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 2) * ((y + 25) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 1) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * ((y + 25) * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(27 * 26 + x) * 4 + i] = sum + (x - 2) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 1) * ((y + 25) * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 25) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 2; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 1) * ((y + 25) * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 26) * 4 + i] = sum;
	float border[] = {
		0, 0, 0, 0
	};
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			border[0] += (y * 5 + x + 3) * (y * 27 + x + 1);
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			border[1] += (y * 5 + x + 2) * (y * 27 + x + 1);
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			border[2] += (y * 5 + x + 1) * (y * 27 + x + 24);
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			border[3] += (y * 5 + x + 1) * (y * 27 + x + 25);
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * (y * 27 + x + 1);
	for (y = 1; y < 26; y++)
	{
		for (i = 0; i < 4; i++)
		{
			c->data.f32[y * 27 * 4 + i] = border[0] + (y - 1) * 27 * (3 + 4 + 5 + 8 + 9 + 10 + 13 + 14 + 15);
			c->data.f32[(y * 27 + 1) * 4 + i] = border[1] + (y - 1) * 27 * (2 + 3 + 4 + 5 + 7 + 8 + 9 + 10 + 12 + 13 + 14 + 15);
			for (x = 2; x < 25; x++)
				c->data.f32[(y * 27 + x) * 4 + i] = sum + ((y - 1) * 27 + x - 2) * 16 * 15 / 2;
			c->data.f32[(y * 27 + 25) * 4 + i] = border[2] + (y - 1) * 27 * (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9 + 11 + 12 + 13 + 14);
			c->data.f32[(y * 27 + 26) * 4 + i] = border[3] + (y - 1) * 27 * (1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13);
		}
	}
	REQUIRE_MATRIX_EQ(b, c, "27x27 matrix should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution with no bias")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_tensor_t* bg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 4, 3, 5, 1);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, bg->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3, 5, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 4 * 3 * 5; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(bg), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 4), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_MATRIX_EQ(b, bg, "convolution with no bias should equal to with bias = 0");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bg);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
}

TEST_CASE("maximum pool network of 55x55 with window of 3x3 and stride of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 55, 55, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_MAX_POOL_FORWARD(3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 113 + y * 110 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("maximum pool network of 57x57 with window of 3x3 and stride of 3")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 57, 57, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 19, 19, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_MAX_POOL_FORWARD(3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(19, 19, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 19; y++)
		for (x = 0; x < 19; x++)
			c->data.f32[y * 19 + x] = 117 + y * 171 + x * 3;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("maximum pool network of 54x54 with window of 2x2 and stride of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 54, 54, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_MAX_POOL_FORWARD(2, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 56 + y * 108 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("average pool network of 55x55 with window of 3x3 and stride of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 55, 55, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_AVERAGE_POOL_FORWARD(3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 57 + y * 110 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("average pool network of 57x57 with window of 3x3 and stride of 3")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 57, 57, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 19, 19, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_AVERAGE_POOL_FORWARD(3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(19, 19, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 19; y++)
		for (x = 0; x < 19; x++)
			c->data.f32[y * 19 + x] = 59 + y * 171 + x * 3;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("average pool network of 54x54 with window of 2x2 and stride of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 54, 54, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 27, 27, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_AVERAGE_POOL_FORWARD(2, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	// configure the inlets.
	int i;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	int x, y;
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 28.5 + y * 108 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 1), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(1, 1, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 3, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 1; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		10., 21., 25., 14.,
		25., 53., 61., 33.,
		37., 77., 85., 45.,
		22., 45., 49., 26.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NHWC(32F, 4, 4, 1), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights and group of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 2), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(2, 2, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 3, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 2; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		1., 10.,  2., 20.,  4., 22.,  3., 12.,
		4., 22.,  9., 45., 13., 49.,  8., 26.,
		10., 28., 21., 57., 25., 61., 14., 32.,
		7., 16., 14., 32., 16., 34.,  9., 18.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NHWC(32F, 4, 4, 2), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights and group of 2, stride of 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 5, 2), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(2, 2, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 2;
	hint.stride.dim[1] = 2;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 3, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 2; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		1., 10.,  2., 11.,  3., 21.,  2., 11.,  3., 12.,
		4., 13.,  5., 14.,  9., 27.,  5., 14.,  6., 15.,
		7., 25.,  9., 27., 17., 53.,  9., 27., 11., 29.,
		4., 13.,  5., 14.,  9., 27.,  5., 14.,  6., 15.,
		7., 16.,  8., 17., 15., 33.,  8., 17.,  9., 18.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NHWC(32F, 5, 5, 2), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights, NCHW")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 4, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(1, 1, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 1, 3, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 1; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		10., 21., 25., 14.,
		25., 53., 61., 33.,
		37., 77., 85., 45.,
		22., 45., 49., 26.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NCHW(32F, 1, 4, 4), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights and group of 2, NCHW")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 4, 4), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(2, 2, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 1, 3, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 2; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		1.,  2.,  4.,  3.,
		4.,  9., 13.,  8.,
		10., 21., 25., 14.,
		7., 14., 16.,  9.,
		10., 20., 22., 12.,
		22., 45., 49., 26.,
		28., 57., 61., 32.,
		16., 32., 34., 18.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NCHW(32F, 2, 4, 4), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolution transpose of 3x3 on 2x2 with given weights and group of 2, stride of 2, NCHW")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 2, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 5, 5), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(2, 2, 0, 3, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 2;
	hint.stride.dim[1] = 2;
	hint.border.begin[0] = 0;
	hint.border.begin[1] = 0;
	hint.border.end[0] = 0;
	hint.border.end[1] = 0;
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 1, 3, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 3 * 3 * 2; i++)
		w->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 2; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 2; i++)
		bias->data.f32[i] = 1;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float cp[] = {
		1.,  2.,  3.,  2.,  3.,
		4.,  5.,  9.,  5.,  6.,
		7.,  9., 17.,  9., 11.,
		4.,  5.,  9.,  5.,  6.,
		7.,  8., 15.,  8.,  9.,
		10., 11., 21., 11., 12.,
		13., 14., 27., 14., 15.,
		25., 27., 53., 27., 29.,
		13., 14., 27., 14., 15.,
		16., 17., 33., 17., 18.
	};
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(cp, CPU_TENSOR_NCHW(32F, 2, 5, 5), 0);
	REQUIRE_TENSOR_EQ(b, c, "convolution transpose output should be exactly the same");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

#include "case_main.h"
