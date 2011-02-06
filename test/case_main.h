#ifndef _GUARD_case_main_h_
#define _GUARD_case_main_h_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <ctype.h>

/* the following functions come from Hans Boehm's conservative gc with slightly
 * modifications, here is the licence:
 * Copyright (c) 1988, 1989 Hans-J. Boehm, Alan J. Demers
 * Copyright (c) 1991-1996 by Xerox Corporation.  All rights reserved.
 * Copyright (c) 1996-1999 by Silicon Graphics.  All rights reserved.
 * Copyright (c) 1999-2004 Hewlett-Packard Development Company, L.P.

 * The file linux_threads.c is also
 * Copyright (c) 1998 by Fergus Henderson.  All rights reserved.

 * The files Makefile.am, and configure.in are
 * Copyright (c) 2001 by Red Hat Inc. All rights reserved.

 * Several files supporting GNU-style builds are copyrighted by the Free
 * Software Foundation, and carry a different license from that given
 * below.

 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY EXPRESSED
 * OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

 * Permission is hereby granted to use or copy this program
 * for any purpose,  provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.

 * A few of the files needed to use the GNU-style build procedure come with
 * slightly different licenses, though they are all similar in spirit.  A few
 * are GPL'ed, but with an exception that should cover all uses in the
 * collector. */

/* Repeatedly perform a read call until the buffer is filled or	*/
/* we encounter EOF.						*/
ssize_t case_repeat_read(int fd, char *buf, size_t count)
{
    ssize_t num_read = 0;
    ssize_t result;
    
    while (num_read < count)
	{
		result = read(fd, buf + num_read, count - num_read);
		if (result < 0)
			return result;
		if (result == 0)
			break;
		num_read += result;
    }
    return num_read;
}

/* Determine the length of a file by incrementally reading it into a buffer */
/* This would be silly to use on a file supporting lseek, but Linux	*/
/* /proc files usually do not.	*/
size_t case_get_file_len(int f)
{
    size_t total = 0;
    ssize_t result;
    char buf[500];

    do {
		result = read(f, buf, 500);
		if (result == -1)
			return 0;
		total += result;
    } while (result > 0);
    return total;
}

size_t case_get_maps_len(void)
{
    int f = open("/proc/self/maps", O_RDONLY);
    size_t result = case_get_file_len(f);
    close(f);
    return result;
}

/*
 * Copy the contents of /proc/self/maps to a buffer in our address space.
 * Return the address of the buffer, or zero on failure.
 * This code could be simplified if we could determine its size
 * ahead of time.
 */
char* case_get_maps()
{
    int f;
    int result;
    static char init_buf[1];
    static char *maps_buf = init_buf;
    static size_t maps_buf_sz = 1;
    size_t maps_size, old_maps_size = 0;

    /* Note that in the presence of threads, the maps file can	*/
    /* essentially shrink asynchronously and unexpectedly as 	*/
    /* threads that we already think of as dead release their	*/
    /* stacks.  And there is no easy way to read the entire	*/
    /* file atomically.  This is arguably a misfeature of the	*/
    /* /proc/.../maps interface.				*/

    /* Since we dont believe the file can grow			*/
    /* asynchronously, it should suffice to first determine	*/
    /* the size (using lseek or read), and then to reread the	*/
    /* file.  If the size is inconsistent we have to retry.	*/
    /* This only matters with threads enabled, and if we use 	*/
    /* this to locate roots (not the default).			*/

    /* Determine the initial size of /proc/self/maps.		*/
    /* Note that lseek doesn't work, at least as of 2.6.15.	*/
	maps_size = case_get_maps_len();
	if (0 == maps_size)
		return 0;

    /* Read /proc/self/maps, growing maps_buf as necessary.	*/
    /* Note that we may not allocate conventionally, and	*/
    /* thus can't use stdio.					*/
	do {
	    while (maps_size >= maps_buf_sz)
		{
			/* Grow only by powers of 2, since we leak "too small" buffers. */
			while (maps_size >= maps_buf_sz)
				maps_buf_sz *= 2;
			maps_buf = malloc(maps_buf_sz);
			/* Recompute initial length, since we allocated.	*/
			/* This can only happen a few times per program		*/
			/* execution.						*/
			maps_size = case_get_maps_len();
			if (0 == maps_size)
				return 0;
	    }
	    f = open("/proc/self/maps", O_RDONLY);
	    if (-1 == f)
			return 0;
		old_maps_size = maps_size;
	    maps_size = 0;
	    do {
			result = case_repeat_read(f, maps_buf, maps_buf_sz - 1);
			if (result <= 0)
				return 0;
			maps_size += result;
	    } while (result == maps_buf_sz - 1);
	    close(f);
	} while (maps_size >= maps_buf_sz || maps_size < old_maps_size);
	/* In the single-threaded case, the second clause is false.	*/
    maps_buf[maps_size] = '\0';
	
    /* Apply fn to result. */
	return maps_buf;
}

//
//  GC_parse_map_entry parses an entry from /proc/self/maps so we can
//  locate all writable data segments that belong to shared libraries.
//  The format of one of these entries and the fields we care about
//  is as follows:
//  XXXXXXXX-XXXXXXXX r-xp 00000000 30:05 260537     name of mapping...\n
//  ^^^^^^^^ ^^^^^^^^ ^^^^          ^^
//  start    end      prot          maj_dev
//
//  Note that since about august 2003 kernels, the columns no longer have
//  fixed offsets on 64-bit kernels.  Hence we no longer rely on fixed offsets
//  anywhere, which is safer anyway.
//

/*
 * Assign various fields of the first line in buf_ptr to *start, *end,
 * *prot, *maj_dev and *mapping_name.  Mapping_name may be NULL.
 * *prot and *mapping_name are assigned pointers into the original
 * buffer.
 */
char* case_parse_map_entry(char* buf_ptr, void** start, void** end, char** prot, unsigned int* maj_dev, char** mapping_name)
{
    char *start_start, *end_start, *maj_dev_start;
    char *p;
    char *endp;

    if (buf_ptr == NULL || *buf_ptr == '\0')
        return NULL;

    p = buf_ptr;
    while (isspace(*p))
		++p;
    start_start = p;
    *start = (void*)strtoul(start_start, &endp, 16);
	p = endp;

    ++p;
    end_start = p;
    *end = (void*)strtoul(end_start, &endp, 16);
	p = endp;

    while (isspace(*p))
		++p;
    *prot = p;
    /* Skip past protection field to offset field */
	while (!isspace(*p))
		++p;
	while (isspace(*p))
		++p;
    /* Skip past offset field, which we ignore */
	while (!isspace(*p))
		++p;
	while (isspace(*p))
		++p;
    maj_dev_start = p;
    *maj_dev = strtoul(maj_dev_start, NULL, 16);

    if (mapping_name == 0)
	{
      while (*p && *p++ != '\n');
    } else {
      while (*p && *p != '\n' && *p != '/' && *p != '[')
		  p++;
      *mapping_name = p;
      while (*p && *p++ != '\n');
    }

    return p;
}

static char __test_header__;

int main(int argc, char** argv)
{
	char* buf = case_get_maps();
	void* start;
	void* end;
	char* prot[4];
	unsigned int maj_dev;
	do {
		buf = case_parse_map_entry(buf, &start, &end, (char**)&prot, &maj_dev, 0);
		if (buf == NULL)
			break;
	} while ((uint64_t)start >= (uint64_t)&__test_header__ || (uint64_t)&__test_header__ >= (uint64_t)end);
	char* start_pointer = (char*)start;
	int total = 0;
#ifndef offsetof
#define offsetof(st, m) __builtin_offsetof(st, m)
#endif
	int len = (uint64_t)end - (uint64_t)start - (sizeof(case_t) - offsetof(case_t, sig));
	int i;
	for (i = 0; i < len; i++)
	{
		case_t* test_suite = (case_t*)(start_pointer + i);
		if (test_suite->sig == 0x883253372849284B)
			total++;
	}
	int j = 1, pass = 0, fail = 0;
	for (i = 0; i < len; i++)
	{
		case_t* test_suite = (case_t*)(start_pointer + i);
		if (test_suite->sig == 0x883253372849284B)
		{
			printf("\033[0;34m[%d/%d]\033[0;30m \033[1;33m[RUN]\033[0;30m %s ...", j, total, test_suite->name);
			int result = 0;
			test_suite->driver(test_suite->name, &result);
			if (result == 0)
			{
				pass++;
				printf("\r\033[0;34m[%d/%d]\033[0;30m \033[1;32m[PASS]\033[0;30m %s    \n", j, total, test_suite->name);
			} else {
				fail++;
				printf("\n\033[0;34m[%d/%d]\033[0;30m \033[1;31m[FAIL]\033[0;30m %s\n", j, total, test_suite->name);
			}
			j++;
		}
	}
	if (fail == 0)
		printf("\033[0;32mall test case(s) passed, congratulations!\033[0;30m\n");
	else
		printf("\033[0;31m%d of %d test case(s) passed\033[0;30m\n", pass, fail + pass);
	return fail;
}

#endif
