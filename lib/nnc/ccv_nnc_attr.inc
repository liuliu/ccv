// For neral networks
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_CONVOLUTION_FORWARD, 0, (1u << 0) | (1u << 1) | (1u << 2), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD, 0, (1u << 0) | (1u << 1) | (1u << 2) | (0 << 3), (1u << 0) | (1u << 1) | (1u << 2),
														  (1u << 0) | (1u << 1) | (0 << 2) | (0 << 3), (0 << 0) | (1u << 1) | (1u << 2))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_MAX_POOL_FORWARD, 0, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_MAX_POOL_BACKWARD, 0, (1u << 0) | (1u << 1) | (1u << 2), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD, 0, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD, 0, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_SOFTMAX_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_SOFTMAX_BACKWARD, CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_BATCH_NORM_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_BATCH_NORM_BACKWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_RELU_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_RELU_BACKWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, (1u << 0) | (1u << 1), 1)
// BLAS
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_GEMM_FORWARD, 0, (1u << 0) | (1u << 1) | (1u << 2), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_GEMM_BACKWARD, 0, (1u << 0) | (1u << 1) | (1u << 2) | (0 << 3), (1u << 0) | (1u << 1) | (1u << 2),
												   (1u << 0) | (1u << 1) | (0 << 2) | (0 << 3), (0 << 0) | (1u << 1) | (1u << 2))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_AXPY_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, (1u << 0) | (1u << 1), 1,
																			 (1u << 0) | (0u << 1), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_AXPY_BACKWARD, CCV_NNC_COMPUTE_ATTR_PASSTHROUGH, 1, (1u << 1) | (1u << 0),
																				  1, (1u << 1) | (0u << 0),
																				  1, (0u << 1) | (1u << 0))
// Element-wise comptation
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWSUM_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1,
																			  (1u << 0) | (1u << 1), 1,
																			  (1u << 0) | (1u << 1) | (1u << 2), 1,
																			  (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWSUM_BACKWARD, CCV_NNC_COMPUTE_ATTR_PASSTHROUGH, 1, 1,
																				   1, (1u << 0) | (1u << 1),
																				   1, (1u << 0) | (1u << 1) | (1u << 2),
																				   1, (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWPROD_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1,
																			   (1u << 0) | (1u << 1), 1,
																			   (1u << 0) | (1u << 1) | (1u << 2), 1,
																			   (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWPROD_BACKWARD, CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES,
	(1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1),
	(1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4), (1u << 0) | (1u << 1) | (1u << 2),
	(1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3),
	(1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5) | (1u << 6), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWDIV_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE | CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES, (1u << 0) | (1u << 1), 1,
																												  (0u << 0) | (1u << 1), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWDIV_BACKWARD, CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES, (1u << 0) | (0u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1),
																					(1u << 0) | (0u << 1) | (1u << 2) | (0u << 3), (1u << 0) | (0u << 1))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWEXP_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWEXP_BACKWARD, CCV_NNC_COMPUTE_ATTR_INPLACE | CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES, (1u << 0) | (0u << 1) | (1u << 2), 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWLOG_FORWARD, CCV_NNC_COMPUTE_ATTR_INPLACE, 1, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_EWLOG_BACKWARD, CCV_NNC_COMPUTE_ATTR_INPLACE | CCV_NNC_COMPUTE_ATTR_NULL_IS_ONES, (1u << 0) | (1u << 1), 1)
// Other transforms
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_SET_FORWARD, 0, 0, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_SET_BACKWARD, 0, 0, 1)
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD, 0, 1, 1,
														   (1u << 0) | (1u << 1), (1u << 0) | (1u << 1),
														   (1u << 0) | (1u << 1) | (1u << 2), (1u << 0) | (1u << 1) | (1u << 2),
														   (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_DATA_TRANSFER_BACKWARD, 0, 1, 1,
															(1u << 0) | (1u << 1), (1u << 0) | (1u << 1),
															(1u << 0) | (1u << 1) | (1u << 2), (1u << 0) | (1u << 1) | (1u << 2),
															(1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD, 0, 1, 1,
															  (1u << 0) | (1u << 1), (1u << 0) | (1u << 1),
															  (1u << 0) | (1u << 1) | (1u << 2), (1u << 0) | (1u << 1) | (1u << 2),
															  (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3))
CCV_NNC_ATTR_DEF(CCV_NNC_COMPUTE_FORMAT_TRANSFORM_BACKWARD, 0, 1, 1,
															   (1u << 0) | (1u << 1), (1u << 0) | (1u << 1),
															   (1u << 0) | (1u << 1) | (1u << 2), (1u << 0) | (1u << 1) | (1u << 2),
															   (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3))
