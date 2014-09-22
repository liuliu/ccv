/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef GUARD_ccv_internal_h
#define GUARD_ccv_internal_h

static int _CCV_PRINT_COUNT __attribute__ ((unused)) = 0;
static int _CCV_PRINT_LOOP __attribute__ ((unused)) = 0;

/* simple utility functions */

#define ccv_descale(x, n) (((x) + (1 << ((n) - 1))) >> (n))
#define conditional_assert(x, expr) if ((x)) { assert(expr); }

#ifdef USE_DISPATCH
#define parallel_for(x, n) dispatch_apply(n, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t x) {
#define parallel_endfor });
#else
#define parallel_for(x, n) { int x; for (x = 0; x < n; x++) {
#define parallel_endfor } }
#endif

/* macro printf utilities */

#define PRINT(l, a, ...) \
	do { \
		if (CCV_CLI_OUTPUT_LEVEL_IS(l)) \
		{ \
			printf(a, ##__VA_ARGS__); \
			fflush(stdout); \
		} \
	} while (0) // using do while (0) to force ; line end


#define FLUSH(l, a, ...) \
	do { \
		if (CCV_CLI_OUTPUT_LEVEL_IS(l)) \
		{ \
			for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
				printf("\b"); \
			for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
				printf(" "); \
			for (_CCV_PRINT_LOOP = 0; _CCV_PRINT_LOOP < _CCV_PRINT_COUNT; _CCV_PRINT_LOOP++) \
				printf("\b"); \
			_CCV_PRINT_COUNT = printf(a, ##__VA_ARGS__); \
			fflush(stdout); \
		} \
	} while (0) // using do while (0) to force ; line end

/* the following macro enables the usage such as:
 * ccv_object_return_if_cached(, db, dc, dd, ...);
 * effectively, it only returns when db, dc and dd are all successfully retrieved from cache */

#define INTERNAL_GARBAGE_TYPEDEF_CONCATENATE_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) \
	(!(_1) || (((int*)(_1))[0] & CCV_GARBAGE)) && (!(_2) || (((int*)(_2))[0] & CCV_GARBAGE)) && (!(_3) || (((int*)(_3))[0] & CCV_GARBAGE)) && (!(_4) || (((int*)(_4))[0] & CCV_GARBAGE)) && (!(_5) || (((int*)(_5))[0] & CCV_GARBAGE)) && (!(_6) || (((int*)(_6))[0] & CCV_GARBAGE)) && (!(_7) || (((int*)(_7))[0] & CCV_GARBAGE)) && (!(_8) || (((int*)(_8))[0] & CCV_GARBAGE)) && (!(_9) || (((int*)(_9))[0] & CCV_GARBAGE)) && (!(_10) || (((int*)(_10))[0] & CCV_GARBAGE)) && (!(_11) || (((int*)(_11))[0] & CCV_GARBAGE)) && (!(_12) || (((int*)(_12))[0] & CCV_GARBAGE)) && (!(_13) || (((int*)(_13))[0] & CCV_GARBAGE)) && (!(_14) || (((int*)(_14))[0] & CCV_GARBAGE)) && (!(_15) || (((int*)(_15))[0] & CCV_GARBAGE)) && (!(_16) || (((int*)(_16))[0] & CCV_GARBAGE)) && (!(_17) || (((int*)(_17))[0] & CCV_GARBAGE)) && (!(_18) || (((int*)(_18))[0] & CCV_GARBAGE)) && (!(_19) || (((int*)(_19))[0] & CCV_GARBAGE)) && (!(_20) || (((int*)(_20))[0] & CCV_GARBAGE)) && (!(_21) || (((int*)(_21))[0] & CCV_GARBAGE)) && (!(_22) || (((int*)(_22))[0] & CCV_GARBAGE)) && (!(_23) || (((int*)(_23))[0] & CCV_GARBAGE)) && (!(_24) || (((int*)(_24))[0] & CCV_GARBAGE)) && (!(_25) || (((int*)(_25))[0] & CCV_GARBAGE)) && (!(_26) || (((int*)(_26))[0] & CCV_GARBAGE)) && (!(_27) || (((int*)(_27))[0] & CCV_GARBAGE)) && (!(_28) || (((int*)(_28))[0] & CCV_GARBAGE)) && (!(_29) || (((int*)(_29))[0] & CCV_GARBAGE)) && (!(_30) || (((int*)(_30))[0] & CCV_GARBAGE)) && (!(_31) || (((int*)(_31))[0] & CCV_GARBAGE)) && (!(_32) || (((int*)(_32))[0] & CCV_GARBAGE)) && (!(_33) || (((int*)(_33))[0] & CCV_GARBAGE)) && (!(_34) || (((int*)(_34))[0] & CCV_GARBAGE)) && (!(_35) || (((int*)(_35))[0] & CCV_GARBAGE)) && (!(_36) || (((int*)(_36))[0] & CCV_GARBAGE)) && (!(_37) || (((int*)(_37))[0] & CCV_GARBAGE)) && (!(_38) || (((int*)(_38))[0] & CCV_GARBAGE)) && (!(_39) || (((int*)(_39))[0] & CCV_GARBAGE)) && (!(_40) || (((int*)(_40))[0] & CCV_GARBAGE)) && (!(_41) || (((int*)(_41))[0] & CCV_GARBAGE)) && (!(_42) || (((int*)(_42))[0] & CCV_GARBAGE)) && (!(_43) || (((int*)(_43))[0] & CCV_GARBAGE)) && (!(_44) || (((int*)(_44))[0] & CCV_GARBAGE)) && (!(_45) || (((int*)(_45))[0] & CCV_GARBAGE)) && (!(_46) || (((int*)(_46))[0] & CCV_GARBAGE)) && (!(_47) || (((int*)(_47))[0] & CCV_GARBAGE)) && (!(_48) || (((int*)(_48))[0] & CCV_GARBAGE)) && (!(_49) || (((int*)(_49))[0] & CCV_GARBAGE)) && (!(_50) || (((int*)(_50))[0] & CCV_GARBAGE)) && (!(_51) || (((int*)(_51))[0] & CCV_GARBAGE)) && (!(_52) || (((int*)(_52))[0] & CCV_GARBAGE)) && (!(_53) || (((int*)(_53))[0] & CCV_GARBAGE)) && (!(_54) || (((int*)(_54))[0] & CCV_GARBAGE)) && (!(_55) || (((int*)(_55))[0] & CCV_GARBAGE)) && (!(_56) || (((int*)(_56))[0] & CCV_GARBAGE)) && (!(_57) || (((int*)(_57))[0] & CCV_GARBAGE)) && (!(_58) || (((int*)(_58))[0] & CCV_GARBAGE)) && (!(_59) || (((int*)(_59))[0] & CCV_GARBAGE)) && (!(_60) || (((int*)(_60))[0] & CCV_GARBAGE)) && (!(_61) || (((int*)(_61))[0] & CCV_GARBAGE)) && (!(_62) || (((int*)(_62))[0] & CCV_GARBAGE)) && (!(_63) || (((int*)(_63))[0] & CCV_GARBAGE))
#define INTERNAL_GARBAGE_TYPEDEF_CONCATENATE(...) INTERNAL_GARBAGE_TYPEDEF_CONCATENATE_N(__VA_ARGS__)

#define INTERNAL_EXPAND_RENEW_MATRIX_LINE_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) \
	(void)((_1) && (((int*)(_1))[0] &= ~CCV_GARBAGE));(void)((_2) && (((int*)(_2))[0] &= ~CCV_GARBAGE));(void)((_3) && (((int*)(_3))[0] &= ~CCV_GARBAGE));(void)((_4) && (((int*)(_4))[0] &= ~CCV_GARBAGE));(void)((_5) && (((int*)(_5))[0] &= ~CCV_GARBAGE));(void)((_6) && (((int*)(_6))[0] &= ~CCV_GARBAGE));(void)((_7) && (((int*)(_7))[0] &= ~CCV_GARBAGE));(void)((_8) && (((int*)(_8))[0] &= ~CCV_GARBAGE));(void)((_9) && (((int*)(_9))[0] &= ~CCV_GARBAGE));(void)((_10) && (((int*)(_10))[0] &= ~CCV_GARBAGE));(void)((_11) && (((int*)(_11))[0] &= ~CCV_GARBAGE));(void)((_12) && (((int*)(_12))[0] &= ~CCV_GARBAGE));(void)((_13) && (((int*)(_13))[0] &= ~CCV_GARBAGE));(void)((_14) && (((int*)(_14))[0] &= ~CCV_GARBAGE));(void)((_15) && (((int*)(_15))[0] &= ~CCV_GARBAGE));(void)((_16) && (((int*)(_16))[0] &= ~CCV_GARBAGE));(void)((_17) && (((int*)(_17))[0] &= ~CCV_GARBAGE));(void)((_18) && (((int*)(_18))[0] &= ~CCV_GARBAGE));(void)((_19) && (((int*)(_19))[0] &= ~CCV_GARBAGE));(void)((_20) && (((int*)(_20))[0] &= ~CCV_GARBAGE));(void)((_21) && (((int*)(_21))[0] &= ~CCV_GARBAGE));(void)((_22) && (((int*)(_22))[0] &= ~CCV_GARBAGE));(void)((_23) && (((int*)(_23))[0] &= ~CCV_GARBAGE));(void)((_24) && (((int*)(_24))[0] &= ~CCV_GARBAGE));(void)((_25) && (((int*)(_25))[0] &= ~CCV_GARBAGE));(void)((_26) && (((int*)(_26))[0] &= ~CCV_GARBAGE));(void)((_27) && (((int*)(_27))[0] &= ~CCV_GARBAGE));(void)((_28) && (((int*)(_28))[0] &= ~CCV_GARBAGE));(void)((_29) && (((int*)(_29))[0] &= ~CCV_GARBAGE));(void)((_30) && (((int*)(_30))[0] &= ~CCV_GARBAGE));(void)((_31) && (((int*)(_31))[0] &= ~CCV_GARBAGE));(void)((_32) && (((int*)(_32))[0] &= ~CCV_GARBAGE));(void)((_33) && (((int*)(_33))[0] &= ~CCV_GARBAGE));(void)((_34) && (((int*)(_34))[0] &= ~CCV_GARBAGE));(void)((_35) && (((int*)(_35))[0] &= ~CCV_GARBAGE));(void)((_36) && (((int*)(_36))[0] &= ~CCV_GARBAGE));(void)((_37) && (((int*)(_37))[0] &= ~CCV_GARBAGE));(void)((_38) && (((int*)(_38))[0] &= ~CCV_GARBAGE));(void)((_39) && (((int*)(_39))[0] &= ~CCV_GARBAGE));(void)((_40) && (((int*)(_40))[0] &= ~CCV_GARBAGE));(void)((_41) && (((int*)(_41))[0] &= ~CCV_GARBAGE));(void)((_42) && (((int*)(_42))[0] &= ~CCV_GARBAGE));(void)((_43) && (((int*)(_43))[0] &= ~CCV_GARBAGE));(void)((_44) && (((int*)(_44))[0] &= ~CCV_GARBAGE));(void)((_45) && (((int*)(_45))[0] &= ~CCV_GARBAGE));(void)((_46) && (((int*)(_46))[0] &= ~CCV_GARBAGE));(void)((_47) && (((int*)(_47))[0] &= ~CCV_GARBAGE));(void)((_48) && (((int*)(_48))[0] &= ~CCV_GARBAGE));(void)((_49) && (((int*)(_49))[0] &= ~CCV_GARBAGE));(void)((_50) && (((int*)(_50))[0] &= ~CCV_GARBAGE));(void)((_51) && (((int*)(_51))[0] &= ~CCV_GARBAGE));(void)((_52) && (((int*)(_52))[0] &= ~CCV_GARBAGE));(void)((_53) && (((int*)(_53))[0] &= ~CCV_GARBAGE));(void)((_54) && (((int*)(_54))[0] &= ~CCV_GARBAGE));(void)((_55) && (((int*)(_55))[0] &= ~CCV_GARBAGE));(void)((_56) && (((int*)(_56))[0] &= ~CCV_GARBAGE));(void)((_57) && (((int*)(_57))[0] &= ~CCV_GARBAGE));(void)((_58) && (((int*)(_58))[0] &= ~CCV_GARBAGE));(void)((_59) && (((int*)(_59))[0] &= ~CCV_GARBAGE));(void)((_60) && (((int*)(_60))[0] &= ~CCV_GARBAGE));(void)((_61) && (((int*)(_61))[0] &= ~CCV_GARBAGE));(void)((_62) && (((int*)(_62))[0] &= ~CCV_GARBAGE));(void)((_63) && (((int*)(_63))[0] &= ~CCV_GARBAGE));
#define INTERNAL_EXPAND_RENEW_MATRIX_LINE(...) INTERNAL_EXPAND_RENEW_MATRIX_LINE_N(__VA_ARGS__)

#define INTERNAL_SEQ_PADDING_ZERO() 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
#define ccv_revive_object_if_cached(...) \
	INTERNAL_EXPAND_RENEW_MATRIX_LINE(__VA_ARGS__, INTERNAL_SEQ_PADDING_ZERO());

#define ccv_object_return_if_cached(rv, ...) { \
	if (INTERNAL_GARBAGE_TYPEDEF_CONCATENATE(__VA_ARGS__, INTERNAL_SEQ_PADDING_ZERO())) { \
		ccv_revive_object_if_cached(__VA_ARGS__); \
		return rv; } }

/* the following 9 lines to generate unique name was taken from Catch: https://github.com/philsquared/Catch
 * here is the licence:
 * Boost Software License - Version 1.0 - August 17th, 2003
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE. */
#ifndef INTERNAL_CATCH_UNIQUE_NAME_LINE2
#define INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line ) name##line
#endif
#ifndef INTERNAL_CATCH_UNIQUE_NAME_LINE
#define INTERNAL_CATCH_UNIQUE_NAME_LINE( name, line ) INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line )
#endif
#ifndef INTERNAL_CATCH_UNIQUE_NAME
#define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __LINE__ )
#endif

#define ccv_sign_with_literal(string) \
	char INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_)[] = (string); \
	size_t INTERNAL_CATCH_UNIQUE_NAME(_ccv_string_size_) = sizeof(INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_));

#define ccv_sign_with_format(size, string, ...) \
	char INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_)[(size)]; \
	memset(INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_), 0, (size)); \
	snprintf(INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_), (size), (string), ##__VA_ARGS__); \
	size_t INTERNAL_CATCH_UNIQUE_NAME(_ccv_string_size_) = (size);

#define CCV_EOF_SIGN ((uint64_t)0)

#define ccv_declare_derived_signature(var, cond, submacro, ...) \
	submacro; \
	uint64_t var = (cond) ? ccv_cache_generate_signature(INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_), INTERNAL_CATCH_UNIQUE_NAME(_ccv_string_size_), __VA_ARGS__) : 0;

/* the following macro enables more finer-control of ccv_declare_derived_signature, notably, it supports much more complex conditions:
 * ccv_declare_derived_signature_case(sig,
 * ccv_sign_with_format(64, "function_name(%f,%f,%f)", a_parameter, b_parameter, c_parameter),
 * ccv_sign_if(% the first custom condition %, a->sig, 0),
 * ccv_sign_if(% the second custom condition %, a->sig, b->sig, 0), ...)
 * the conditions will pass on, thus, there is no early termination, if the last condition meets, the signature will be determined
 * by the operation associated with the last condition */

#define ccv_sign_if(cond, ...) \
	if (cond) { \
		INTERNAL_CATCH_UNIQUE_NAME(_ccv_temp_sig_) = ccv_cache_generate_signature(INTERNAL_CATCH_UNIQUE_NAME(_ccv_identifier_), INTERNAL_CATCH_UNIQUE_NAME(_ccv_string_size_), __VA_ARGS__); \
	}

#define INTERNAL_EXPAND_MACRO_ARGUMENT_TO_LINE_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) \
	_1;_2;_3;_4;_5;_6;_7;_8;_9;_10;_11;_12;_13;_14;_15;_16;_17;_18;_19;_20;_21;_22;_23;_24;_25;_26;_27;_28;_29;_30;_31;_32;_33;_34;_35;_36;_37;_38;_39;_40;_41;_42;_43;_44;_45;_46;_47;_48;_49;_50;_51;_52;_53;_54;_55;_56;_57;_58;_59;_60;_61;_62;_63
#define INTERNAL_EXPAND_MACRO_ARGUMENT_TO_LINE(...) INTERNAL_EXPAND_MACRO_ARGUMENT_TO_LINE_N(__VA_ARGS__)

#define INTERNAL_SEQ_PADDING_LINE() ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

#define ccv_declare_derived_signature_case(var, submacro, ...) \
	submacro; \
	uint64_t INTERNAL_CATCH_UNIQUE_NAME(_ccv_temp_sig_) = 0; \
	INTERNAL_EXPAND_MACRO_ARGUMENT_TO_LINE(__VA_ARGS__, INTERNAL_SEQ_PADDING_LINE()); \
	uint64_t var = INTERNAL_CATCH_UNIQUE_NAME(_ccv_temp_sig_);

/* the macros enable us to preserve state of the program at any point in a structure way, this is borrowed from coroutine idea */

#define ccv_function_state_reserve_field int line_no;
#define ccv_function_state_begin(reader, s, file) (reader)((file), &(s)); switch ((s).line_no) { case 0:;
#define ccv_function_state_resume(writer, s, file) do { (s).line_no = __LINE__; (writer)(&(s), (file)); case __LINE__:; } while (0)
#define ccv_function_state_finish() }

/* the factor used to provide higher accuracy in integer type (all integer computation in some cases) */
#define _ccv_get_32s_value(ptr, i, factor) (((int*)(ptr))[(i)] << factor)
#define _ccv_get_32f_value(ptr, i, factor) ((float*)(ptr))[(i)]
#define _ccv_get_64s_value(ptr, i, factor) (((int64_t*)(ptr))[(i)] << factor)
#define _ccv_get_64f_value(ptr, i, factor) ((double*)(ptr))[(i)]
#define _ccv_get_8u_value(ptr, i, factor) (((unsigned char*)(ptr))[(i)] << factor)

#define ccv_matrix_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_getter_integer_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_8U: { block(__VA_ARGS__, _ccv_get_8u_value); break; } \
	default: { assert((type & CCV_32S) || (type & CCV_64S) || (type & CCV_8U)); } } }

#define ccv_matrix_getter_integer_only_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_8U: { block(__VA_ARGS__, _ccv_get_8u_value); break; } \
	default: { assert((type & CCV_32S) || (type & CCV_64S) || (type & CCV_8U)); } } }

#define ccv_matrix_getter_integer_only_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_get_64s_value); break; } \
	case CCV_8U: { block(__VA_ARGS__, _ccv_get_8u_value); break; } \
	default: { assert((type & CCV_32S) || (type & CCV_64S) || (type & CCV_8U)); } } }

#define ccv_matrix_getter_float_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { assert((type & CCV_32F) || (type & CCV_64F)); } } }

#define ccv_matrix_typeof(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define ccv_matrix_typeof_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define ccv_matrix_typeof_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define _ccv_set_32s_value(ptr, i, value, factor) (((int*)(ptr))[(i)] = (int)(value) >> factor)
#define _ccv_set_32f_value(ptr, i, value, factor) (((float*)(ptr))[(i)] = (float)(value))
#define _ccv_set_64s_value(ptr, i, value, factor) (((int64_t*)(ptr))[(i)] = (int64_t)(value) >> factor)
#define _ccv_set_64f_value(ptr, i, value, factor) (((double*)(ptr))[(i)] = (double)(value))
#define _ccv_set_8u_value(ptr, i, value, factor) (((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value) >> factor, 0, 255))

#define ccv_matrix_setter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_integer_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value); break; } \
	case CCV_8U: { block(__VA_ARGS__, _ccv_set_8u_value); break; } \
	default: { assert((type & CCV_32S) || (type & CCV_64S) || (type & CCV_8U)); } } }

#define ccv_matrix_setter_float_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { assert((type & CCV_32F) || (type & CCV_64F)); } } }

#define ccv_matrix_setter_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_setter_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_setter_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_setter_getter_integer_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_8U: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); break; } \
	default: { assert((type & CCV_32S) || (type & CCV_64S) || (type & CCV_8U)); } } }

#define ccv_matrix_setter_getter_float_only(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { assert((type & CCV_32F) || (type & CCV_64F)); } } }

#define ccv_matrix_typeof_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64S: { block(__VA_ARGS__, int64_t, _ccv_set_64s_value, _ccv_get_64s_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

/****************************************************************************************\

  Generic implementation of QuickSort algorithm.
  ----------------------------------------------
  Using this macro user can declare customized sort function that can be much faster
  than built-in qsort function because of lower overhead on elements
  comparison and exchange. The macro takes less_than (or LT) argument - a macro or function
  that takes 2 arguments returns non-zero if the first argument should be before the second
  one in the sorted sequence and zero otherwise.

  Example:

    Suppose that the task is to sort points by ascending of y coordinates and if
    y's are equal x's should ascend.

    The code is:
    ------------------------------------------------------------------------------
           #define cmp_pts( pt1, pt2 ) \
               ((pt1).y < (pt2).y || ((pt1).y < (pt2).y && (pt1).x < (pt2).x))

           [static] CV_IMPLEMENT_QSORT( icvSortPoints, CvPoint, cmp_pts )
    ------------------------------------------------------------------------------

    After that the function "void icvSortPoints( CvPoint* array, size_t total, int aux );"
    is available to user.

  aux is an additional parameter, which can be used when comparing elements.
  The current implementation was derived from *BSD system qsort():

    * Copyright (c) 1992, 1993
    *  The Regents of the University of California.  All rights reserved.
    *
    * Redistribution and use in source and binary forms, with or without
    * modification, are permitted provided that the following conditions
    * are met:
    * 1. Redistributions of source code must retain the above copyright
    *    notice, this list of conditions and the following disclaimer.
    * 2. Redistributions in binary form must reproduce the above copyright
    *    notice, this list of conditions and the following disclaimer in the
    *    documentation and/or other materials provided with the distribution.
    * 3. All advertising materials mentioning features or use of this software
    *    must display the following acknowledgement:
    *  This product includes software developed by the University of
    *  California, Berkeley and its contributors.
    * 4. Neither the name of the University nor the names of its contributors
    *    may be used to endorse or promote products derived from this software
    *    without specific prior written permission.
    *
    * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
    * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
    * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    * SUCH DAMAGE.

\****************************************************************************************/

#define CCV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

#define CCV_IMPLEMENT_QSORT_EX(func_name, T, LT, swap_func, user_data_type)                     \
void func_name(T *array, size_t total, user_data_type aux)                                      \
{                                                                                               \
    int isort_thresh = 7;                                                                       \
    T t;                                                                                        \
    int sp = 0;                                                                                 \
                                                                                                \
    struct                                                                                      \
    {                                                                                           \
        T *lb;                                                                                  \
        T *ub;                                                                                  \
    }                                                                                           \
    stack[48];                                                                                  \
                                                                                                \
    if( total <= 1 )                                                                            \
        return;                                                                                 \
                                                                                                \
    stack[0].lb = array;                                                                        \
    stack[0].ub = array + (total - 1);                                                          \
                                                                                                \
    while( sp >= 0 )                                                                            \
    {                                                                                           \
        T* left = stack[sp].lb;                                                                 \
        T* right = stack[sp--].ub;                                                              \
                                                                                                \
        for(;;)                                                                                 \
        {                                                                                       \
            int i, n = (int)(right - left) + 1, m;                                              \
            T* ptr;                                                                             \
            T* ptr2;                                                                            \
                                                                                                \
            if( n <= isort_thresh )                                                             \
            {                                                                                   \
            insert_sort:                                                                        \
                for( ptr = left + 1; ptr <= right; ptr++ )                                      \
                {                                                                               \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1], aux); ptr2--)          \
                        swap_func( ptr2[0], ptr2[-1], array, aux, t );                          \
                }                                                                               \
                break;                                                                          \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                T* left0;                                                                       \
                T* left1;                                                                       \
                T* right0;                                                                      \
                T* right1;                                                                      \
                T* pivot;                                                                       \
                T* a;                                                                           \
                T* b;                                                                           \
                T* c;                                                                           \
                int swap_cnt = 0;                                                               \
                                                                                                \
                left0 = left;                                                                   \
                right0 = right;                                                                 \
                pivot = left + (n/2);                                                           \
                                                                                                \
                if( n > 40 )                                                                    \
                {                                                                               \
                    int d = n / 8;                                                              \
                    a = left, b = left + d, c = left + 2*d;                                     \
                    left = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))  \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = pivot - d, b = pivot, c = pivot + d;                                    \
                    pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = right - 2*d, b = right - d, c = right;                                  \
                    right = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                }                                                                               \
                                                                                                \
                a = left, b = pivot, c = right;                                                 \
                pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))     \
                                   : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));         \
                if( pivot != left0 )                                                            \
                {                                                                               \
                    swap_func( *pivot, *left0, array, aux, t );                                 \
                    pivot = left0;                                                              \
                }                                                                               \
                left = left1 = left0 + 1;                                                       \
                right = right1 = right0;                                                        \
                                                                                                \
                for(;;)                                                                         \
                {                                                                               \
                    while( left <= right && !LT(*pivot, *left, aux) )                           \
                    {                                                                           \
                        if( !LT(*left, *pivot, aux) )                                           \
                        {                                                                       \
                            if( left > left1 )                                                  \
                                swap_func( *left1, *left, array, aux, t );                      \
                            swap_cnt = 1;                                                       \
                            left1++;                                                            \
                        }                                                                       \
                        left++;                                                                 \
                    }                                                                           \
                                                                                                \
                    while( left <= right && !LT(*right, *pivot, aux) )                          \
                    {                                                                           \
                        if( !LT(*pivot, *right, aux) )                                          \
                        {                                                                       \
                            if( right < right1 )                                                \
                                swap_func( *right1, *right, array, aux, t );                    \
                            swap_cnt = 1;                                                       \
                            right1--;                                                           \
                        }                                                                       \
                        right--;                                                                \
                    }                                                                           \
                                                                                                \
                    if( left > right )                                                          \
                        break;                                                                  \
                    swap_func( *left, *right, array, aux, t );                                  \
                    swap_cnt = 1;                                                               \
                    left++;                                                                     \
                    right--;                                                                    \
                }                                                                               \
                                                                                                \
                if( swap_cnt == 0 )                                                             \
                {                                                                               \
                    left = left0, right = right0;                                               \
                    goto insert_sort;                                                           \
                }                                                                               \
                                                                                                \
                n = ccv_min( (int)(left1 - left0), (int)(left - left1) );                       \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left0[i], left[i-n], array, aux, t );                            \
                                                                                                \
                n = ccv_min( (int)(right0 - right1), (int)(right1 - right) );                   \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left[i], right0[i-n+1], array, aux, t );                         \
                n = (int)(left - left1);                                                        \
                m = (int)(right1 - right);                                                      \
                if( n > 1 )                                                                     \
                {                                                                               \
                    if( m > 1 )                                                                 \
                    {                                                                           \
                        if( n > m )                                                             \
                        {                                                                       \
                            stack[++sp].lb = left0;                                             \
                            stack[sp].ub = left0 + n - 1;                                       \
                            left = right0 - m + 1, right = right0;                              \
                        }                                                                       \
                        else                                                                    \
                        {                                                                       \
                            stack[++sp].lb = right0 - m + 1;                                    \
                            stack[sp].ub = right0;                                              \
                            left = left0, right = left0 + n - 1;                                \
                        }                                                                       \
                    }                                                                           \
                    else                                                                        \
                        left = left0, right = left0 + n - 1;                                    \
                }                                                                               \
                else if( m > 1 )                                                                \
                    left = right0 - m + 1, right = right0;                                      \
                else                                                                            \
                    break;                                                                      \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
}

#define _ccv_qsort_default_swap(a, b, array, aux, t) CCV_SWAP((a), (b), (t))

#define CCV_IMPLEMENT_QSORT(func_name, T, cmp) \
    CCV_IMPLEMENT_QSORT_EX(func_name, T, cmp, _ccv_qsort_default_swap, int)

#define CCV_IMPLEMENT_MEDIAN(func_name, T) \
T func_name(T* buf, int low, int high) \
{                                                    \
	T w;                                             \
	int middle, ll, hh;                              \
	int median = (low + high) / 2;                   \
	for (;;)                                         \
	{                                                \
		if (high <= low)                             \
			return buf[median];                      \
		if (high == low + 1)                         \
		{                                            \
			if (buf[low] > buf[high])                \
				CCV_SWAP(buf[low], buf[high], w);    \
			return buf[median];                      \
		}                                            \
		middle = (low + high) / 2;                   \
		if (buf[middle] > buf[high])                 \
			CCV_SWAP(buf[middle], buf[high], w);     \
		if (buf[low] > buf[high])                    \
			CCV_SWAP(buf[low], buf[high], w);        \
		if (buf[middle] > buf[low])                  \
			CCV_SWAP(buf[middle], buf[low], w);      \
		CCV_SWAP(buf[middle], buf[low + 1], w);      \
		ll = low + 1;                                \
		hh = high;                                   \
		for (;;)                                     \
		{                                            \
			do ll++; while (buf[low] > buf[ll]);     \
			do hh--; while (buf[hh] > buf[low]);     \
			if (hh < ll)                             \
				break;                               \
			CCV_SWAP(buf[ll], buf[hh], w);           \
		}                                            \
		CCV_SWAP(buf[low], buf[hh], w);              \
		if (hh <= median)                            \
			low = ll;                                \
		else if (hh >= median)                       \
			high = hh - 1;                           \
	}                                                \
}

#endif
