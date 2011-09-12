/*
 * Definitions.hpp
 *
 *  Created on: 17/06/2009
 *      Author: rcamargo
 */

#ifndef DEFINITIONS_HPP_
#define DEFINITIONS_HPP_

//#include <time.h>
#include <stdlib.h>

#define PI 3.14159
typedef float ftype;
//typedef double ftype;

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
	typedef unsigned __int64 uint64;
	typedef unsigned short ucomp;
#elif defined(__APPLE__)
    typedef unsigned long long uint64;
    typedef unsigned short ucomp;
    struct random_data {};
    void random_r( struct random_data *buf, int32_t *del );
    void initstate_r(unsigned int seed, char *statebuf, size_t statelen, struct random_data *buf);
#else
    typedef unsigned long long uint64;
    typedef unsigned short ucomp;
#endif

#endif /* DEFINITIONS_HPP_ */
