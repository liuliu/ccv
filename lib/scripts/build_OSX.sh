#!/bin/sh
# build.sh

# Created by Florian Denis http://floriandenis.net
# CCV library : https://github.com/liuliu/ccv
# Usage: in the `lib` directory, ./scripts/build_OSX.sh

OUTDIR="./outdir"

# Define the archs and relevant sdk we want to build for
archs=( 
        "i386","macosx" 
        "x86_64","macosx"
        )
        
# Use like `setenv "armv7" "iphoneos"`
setenv()
{
    unset SDK SDKROOT CFLAGS CPPFLAGS CXXFLAGS LDFLAGS CC CXX LD AR AS NM RANLIB
    
    export SDK=$2
    
    export SDKROOT=`xcrun -find -sdk $SDK --show-sdk-path`

    export CFLAGS="-arch $1 -isysroot $SDKROOT -I$SDKROOT/usr/include/ -D HAVE_ACCELERATE_FRAMEWORK -D USE_DISPATCH -fblocks"
    export CPPFLAGS=$CFLAGS
    export CXXFLAGS=$CFLAGS

    export LDFLAGS="-lm -framework Accelerate -ldispatch -lBlocksRuntime  -L$SDKROOT/usr/lib/"

    export CC=`xcrun -find -sdk $SDK clang`
    export CXX=`xcrun -find -sdk $SDK clang++`
    
    export LD=`xcrun -find -sdk $SDK ld` 
    export AR=`xcrun -find -sdk $SDK ar` 
    export AS=`xcrun -find -sdk $SDK as` 
    export NM=`xcrun -find -sdk $SDK nm` 
    export RANLIB=`xcrun -find -sdk $SDK ranlib` 

}

cleanup()
{
    make clean 2> /dev/null
    make distclean 2> /dev/null
}


# Use like `build "armv7" "iphoneos"`
build()
{
    cleanup
    
    # Set the proper env var
    setenv $1 $2
    echo "clang" > .CC
    echo $CFLAGS > .DEF
    echo $LDFLAGS > .LN
    
    # Build
    make all
    
}

create_fat_binary()
{
    LIPO_CMD="lipo"
    OLDIFS=$IFS; IFS=','
    for i in "${archs[@]}"; do
        set $i
        lib_path="$OUTDIR/$1/libccv.a"
        LIPO_CMD="$LIPO_CMD -arch $1 $lib_path"
    done
    IFS=$OLDIFS
    LIPO_CMD="$LIPO_CMD -create -output $OUTDIR/lib/libccv.a"
    $LIPO_CMD
}


#######################
# CCV Lib
#######################

# Clean up
rm -rf $OUTDIR

# Loop over the archs
OLDIFS=$IFS; IFS=','
for i in "${archs[@]}"
do
    set $i
    
    # Create destination folder
    mkdir -p $OUTDIR/$1
    
    # Build the libccv.a
    build $1 $2
    
    # Copy to destination directory
    cp -rvf libccv.a $OUTDIR/$1
    
done
IFS=$OLDIFS

# Clean the last build
cleanup

# Copy headers
mkdir -p "$OUTDIR/include" && cp -vf ccv.h "$OUTDIR/include"

# Build fat binary
mkdir -p "$OUTDIR/lib" && create_fat_binary

# Remove intermediate builds
OLDIFS=$IFS; IFS=','
for i in "${archs[@]}"; do
    set $i
    rm -rf $OUTDIR/$1
done
IFS=$OLDIFS

echo "Finished!"
open $OUTDIR