#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "3rdparty/sqlite3/sqlite3.h"
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("zero out a tensor")
{
	const ccv_nnc_tensor_param_t params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = {
			10, 20, 30, 4, 5, 6,
		},
	};
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, params, 0);
	int i;
	for (i = 0; i < 10 * 20 * 30 * 4 * 5 * 6; i++)
		tensor->data.f32[i] = 1;
	ccv_nnc_tensor_zero(tensor);
	for (i = 0; i < 10 * 20 * 30 * 4 * 5 * 6; i++)
		REQUIRE_EQ(0, tensor->data.f32[i], "should be zero'ed at %d", i);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("zero out a tensor view")
{
	const ccv_nnc_tensor_param_t params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = {
			10, 20, 30, 4, 5, 6,
		},
	};
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, params, 0);
	int c;
	for (c = 0; c < 10 * 20 * 30 * 4 * 5 * 6; c++)
		a_tensor->data.f32[c] = 1;
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {
		1, 2, 5, 1, 1, 1,
	};
	const ccv_nnc_tensor_param_t new_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = {
			8, 12, 15, 2, 3, 4,
		},
	};
	ccv_nnc_tensor_view_t a_tensor_view = ccv_nnc_tensor_view(a_tensor, new_params, ofs, DIM_ALLOC(20 * 30 * 4 * 5 * 6, 30 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1));
	ccv_nnc_tensor_zero(&a_tensor_view);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_new(0, params, 0);
	for (c = 0; c < 10 * 20 * 30 * 4 * 5 * 6; c++)
		b_tensor->data.f32[c] = 1;
	ccv_nnc_tensor_view_t b_tensor_view = ccv_nnc_tensor_view(b_tensor, new_params, ofs, DIM_ALLOC(20 * 30 * 4 * 5 * 6, 30 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1));
	int i[6];
	float* tvp[6];
	tvp[5] = b_tensor_view.data.f32;
	for (i[5] = 0; i[5] < b_tensor_view.info.dim[0]; i[5]++)
	{
		tvp[4] = tvp[5];
		for (i[4] = 0; i[4] < b_tensor_view.info.dim[1]; i[4]++)
		{
			tvp[3] = tvp[4];
			for (i[3] = 0; i[3] < b_tensor_view.info.dim[2]; i[3]++)
			{
				tvp[2] = tvp[3];
				for (i[2] = 0; i[2] < b_tensor_view.info.dim[3]; i[2]++)
				{
					tvp[1] = tvp[2];
					for (i[1] = 0; i[1] < b_tensor_view.info.dim[4]; i[1]++)
					{
						tvp[0] = tvp[1];
						for (i[0] = 0; i[0] < b_tensor_view.info.dim[5]; i[0]++)
						{
							tvp[0][i[0]] = 0;
						}
						tvp[1] += b_tensor_view.stride[4];
					}
					tvp[2] += b_tensor_view.stride[3];
				}
				tvp[3] += b_tensor_view.stride[2];
			}
			tvp[4] += b_tensor_view.stride[1];
		}
		tvp[5] += b_tensor_view.stride[0];
	}
	REQUIRE_TENSOR_EQ(a_tensor, b_tensor, "zero'ed tensor view should be equal");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b_tensor);
}

TEST_CASE("hint tensor")
{
	ccv_nnc_tensor_param_t a = CPU_TENSOR_NHWC(32F, 234, 128, 3);
	ccv_nnc_hint_t hint = {
		.border = {
			.begin = {1, 1},
			.end = {1, 2}
		},
		.stride = {
			.dim = {8, 7}
		}
	};
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 4, 5, 3);
	ccv_nnc_tensor_param_t b;
	ccv_nnc_tensor_param_t w = CPU_TENSOR_NHWC(32F, 128, 4, 5, 3);
	ccv_nnc_tensor_param_t bias = CPU_TENSOR_NHWC(32F, 128);
	ccv_nnc_hint_tensor_auto(cmd, TENSOR_PARAM_LIST(a, w, bias), hint, &b, 1);
	REQUIRE_EQ(b.dim[0], 30, "height should be 30");
	REQUIRE_EQ(b.dim[1], 19, "width should be 19");
	REQUIRE_EQ(b.dim[2], 128, "channel should be the convolution filter count");
}

TEST_CASE("tensor persistence")
{
	sqlite3* handle;
	sqlite3_open("tensors.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_write(tensor, handle, "x");
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = 0;
	ccv_nnc_tensor_read(handle, "x", 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "x", 0, &tensor2);
	sqlite3_close(handle);
	REQUIRE_TENSOR_EQ(tensor1, tensor, "the first tensor should equal to the second");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2->data.f32, tensor->data.f32, 10, 1e-5, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("resize tensor")
{
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 12, 3), 0);
	int i;
	for (i = 0; i < 12 * 12 * 3; i++)
		tensor->data.f32[i] = i;
	tensor = ccv_nnc_tensor_resize(tensor, CPU_TENSOR_NHWC(32F, 23, 23, 3));
	for (i = 12 * 12 * 3; i < 23 * 23 * 3; i++)
		tensor->data.f32[i] = i;
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 23, 23, 3), 0);
	for (i = 0; i < 23 * 23 * 3; i++)
		b->data.f32[i] = i;
	REQUIRE_TENSOR_EQ(tensor, b, "should retain the content when resize a tensor");
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format 5-d tensor into string")
{
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 5, 2, 9, 9), 0);
	int i;
	for (i = 0; i < 81 * 2 * 5 * 2; i++)
		tensor->data.i32[i] = i;
	char* str = ccv_nnc_tensor_format_new(tensor);
	const char t[] = "[\n"
"  [[[[         0,          1,          2,  ...,          6,          7,          8],\n"
"     [         9,         10,         11,  ...,         15,         16,         17],\n"
"     [        18,         19,         20,  ...,         24,         25,         26],\n"
"     ...,\n"
"     [        54,         55,         56,  ...,         60,         61,         62],\n"
"     [        63,         64,         65,  ...,         69,         70,         71],\n"
"     [        72,         73,         74,  ...,         78,         79,         80]],\n"
"    [[        81,         82,         83,  ...,         87,         88,         89],\n"
"     [        90,         91,         92,  ...,         96,         97,         98],\n"
"     [        99,        100,        101,  ...,        105,        106,        107],\n"
"     ...,\n"
"     [       135,        136,        137,  ...,        141,        142,        143],\n"
"     [       144,        145,        146,  ...,        150,        151,        152],\n"
"     [       153,        154,        155,  ...,        159,        160,        161]]],\n"
"   [[[       162,        163,        164,  ...,        168,        169,        170],\n"
"     [       171,        172,        173,  ...,        177,        178,        179],\n"
"     [       180,        181,        182,  ...,        186,        187,        188],\n"
"     ...,\n"
"     [       216,        217,        218,  ...,        222,        223,        224],\n"
"     [       225,        226,        227,  ...,        231,        232,        233],\n"
"     [       234,        235,        236,  ...,        240,        241,        242]],\n"
"    [[       243,        244,        245,  ...,        249,        250,        251],\n"
"     [       252,        253,        254,  ...,        258,        259,        260],\n"
"     [       261,        262,        263,  ...,        267,        268,        269],\n"
"     ...,\n"
"     [       297,        298,        299,  ...,        303,        304,        305],\n"
"     [       306,        307,        308,  ...,        312,        313,        314],\n"
"     [       315,        316,        317,  ...,        321,        322,        323]]],\n"
"   ...,\n"
"   [[[       486,        487,        488,  ...,        492,        493,        494],\n"
"     [       495,        496,        497,  ...,        501,        502,        503],\n"
"     [       504,        505,        506,  ...,        510,        511,        512],\n"
"     ...,\n"
"     [       540,        541,        542,  ...,        546,        547,        548],\n"
"     [       549,        550,        551,  ...,        555,        556,        557],\n"
"     [       558,        559,        560,  ...,        564,        565,        566]],\n"
"    [[       567,        568,        569,  ...,        573,        574,        575],\n"
"     [       576,        577,        578,  ...,        582,        583,        584],\n"
"     [       585,        586,        587,  ...,        591,        592,        593],\n"
"     ...,\n"
"     [       621,        622,        623,  ...,        627,        628,        629],\n"
"     [       630,        631,        632,  ...,        636,        637,        638],\n"
"     [       639,        640,        641,  ...,        645,        646,        647]]],\n"
"   [[[       648,        649,        650,  ...,        654,        655,        656],\n"
"     [       657,        658,        659,  ...,        663,        664,        665],\n"
"     [       666,        667,        668,  ...,        672,        673,        674],\n"
"     ...,\n"
"     [       702,        703,        704,  ...,        708,        709,        710],\n"
"     [       711,        712,        713,  ...,        717,        718,        719],\n"
"     [       720,        721,        722,  ...,        726,        727,        728]],\n"
"    [[       729,        730,        731,  ...,        735,        736,        737],\n"
"     [       738,        739,        740,  ...,        744,        745,        746],\n"
"     [       747,        748,        749,  ...,        753,        754,        755],\n"
"     ...,\n"
"     [       783,        784,        785,  ...,        789,        790,        791],\n"
"     [       792,        793,        794,  ...,        798,        799,        800],\n"
"     [       801,        802,        803,  ...,        807,        808,        809]]]],\n"
"  [[[[       810,        811,        812,  ...,        816,        817,        818],\n"
"     [       819,        820,        821,  ...,        825,        826,        827],\n"
"     [       828,        829,        830,  ...,        834,        835,        836],\n"
"     ...,\n"
"     [       864,        865,        866,  ...,        870,        871,        872],\n"
"     [       873,        874,        875,  ...,        879,        880,        881],\n"
"     [       882,        883,        884,  ...,        888,        889,        890]],\n"
"    [[       891,        892,        893,  ...,        897,        898,        899],\n"
"     [       900,        901,        902,  ...,        906,        907,        908],\n"
"     [       909,        910,        911,  ...,        915,        916,        917],\n"
"     ...,\n"
"     [       945,        946,        947,  ...,        951,        952,        953],\n"
"     [       954,        955,        956,  ...,        960,        961,        962],\n"
"     [       963,        964,        965,  ...,        969,        970,        971]]],\n"
"   [[[       972,        973,        974,  ...,        978,        979,        980],\n"
"     [       981,        982,        983,  ...,        987,        988,        989],\n"
"     [       990,        991,        992,  ...,        996,        997,        998],\n"
"     ...,\n"
"     [      1026,       1027,       1028,  ...,       1032,       1033,       1034],\n"
"     [      1035,       1036,       1037,  ...,       1041,       1042,       1043],\n"
"     [      1044,       1045,       1046,  ...,       1050,       1051,       1052]],\n"
"    [[      1053,       1054,       1055,  ...,       1059,       1060,       1061],\n"
"     [      1062,       1063,       1064,  ...,       1068,       1069,       1070],\n"
"     [      1071,       1072,       1073,  ...,       1077,       1078,       1079],\n"
"     ...,\n"
"     [      1107,       1108,       1109,  ...,       1113,       1114,       1115],\n"
"     [      1116,       1117,       1118,  ...,       1122,       1123,       1124],\n"
"     [      1125,       1126,       1127,  ...,       1131,       1132,       1133]]],\n"
"   ...,\n"
"   [[[      1296,       1297,       1298,  ...,       1302,       1303,       1304],\n"
"     [      1305,       1306,       1307,  ...,       1311,       1312,       1313],\n"
"     [      1314,       1315,       1316,  ...,       1320,       1321,       1322],\n"
"     ...,\n"
"     [      1350,       1351,       1352,  ...,       1356,       1357,       1358],\n"
"     [      1359,       1360,       1361,  ...,       1365,       1366,       1367],\n"
"     [      1368,       1369,       1370,  ...,       1374,       1375,       1376]],\n"
"    [[      1377,       1378,       1379,  ...,       1383,       1384,       1385],\n"
"     [      1386,       1387,       1388,  ...,       1392,       1393,       1394],\n"
"     [      1395,       1396,       1397,  ...,       1401,       1402,       1403],\n"
"     ...,\n"
"     [      1431,       1432,       1433,  ...,       1437,       1438,       1439],\n"
"     [      1440,       1441,       1442,  ...,       1446,       1447,       1448],\n"
"     [      1449,       1450,       1451,  ...,       1455,       1456,       1457]]],\n"
"   [[[      1458,       1459,       1460,  ...,       1464,       1465,       1466],\n"
"     [      1467,       1468,       1469,  ...,       1473,       1474,       1475],\n"
"     [      1476,       1477,       1478,  ...,       1482,       1483,       1484],\n"
"     ...,\n"
"     [      1512,       1513,       1514,  ...,       1518,       1519,       1520],\n"
"     [      1521,       1522,       1523,  ...,       1527,       1528,       1529],\n"
"     [      1530,       1531,       1532,  ...,       1536,       1537,       1538]],\n"
"    [[      1539,       1540,       1541,  ...,       1545,       1546,       1547],\n"
"     [      1548,       1549,       1550,  ...,       1554,       1555,       1556],\n"
"     [      1557,       1558,       1559,  ...,       1563,       1564,       1565],\n"
"     ...,\n"
"     [      1593,       1594,       1595,  ...,       1599,       1600,       1601],\n"
"     [      1602,       1603,       1604,  ...,       1608,       1609,       1610],\n"
"     [      1611,       1612,       1613,  ...,       1617,       1618,       1619]]]]\n"
"]";
	REQUIRE(memcmp(str, t, strlen(t) + 1) == 0, "output should be equal");
	ccfree(str);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("format small 2-d tensor into string")
{
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4, 4), 0);
	int i;
	for (i = 0; i < 4 * 4; i++)
		tensor->data.i32[i] = i;
	char* str = ccv_nnc_tensor_format_new(tensor);
	const char t[] = "[\n"
"  [         0,          1,          2,          3],\n"
"  [         4,          5,          6,          7],\n"
"  [         8,          9,         10,         11],\n"
"  [        12,         13,         14,         15]\n"
"]";
	REQUIRE(memcmp(str, t, strlen(t) + 1) == 0, "output should be equal");
	ccfree(str);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("format small 1-d tensor into string")
{
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 12), 0);
	int i;
	for (i = 0; i < 12; i++)
		tensor->data.i32[i] = i;
	char* str = ccv_nnc_tensor_format_new(tensor);
	const char t[] = "[\n"
"           0,          1,          2,          3,          4,          5,          6,          7,\n"
"           8,          9,         10,         11\n"
"]";
	REQUIRE(memcmp(str, t, strlen(t) + 1) == 0, "output should be equal");
	ccfree(str);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("format large 1-d tensor into string")
{
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 68), 0);
	int i;
	for (i = 0; i < 68; i++)
		tensor->data.i32[i] = i;
	char* str = ccv_nnc_tensor_format_new(tensor);
	const char t[] = "[\n"
"           0,          1,          2,          3,          4,          5,          6,          7,\n"
"           8,          9,         10,         11,         12,         13,         14,         15,\n"
"          16,         17,         18,         19,         20,         21,         22,         23,\n"
"  ...,\n"
"          48,         49,         50,         51,         52,         53,         54,         55,\n"
"          56,         57,         58,         59,         60,         61,         62,         63,\n"
"          64,         65,         66,         67\n"
"]";
	REQUIRE(memcmp(str, t, strlen(t) + 1) == 0, "output should be equal");
	ccfree(str);
	ccv_nnc_tensor_free(tensor);
}

#include "case_main.h"
