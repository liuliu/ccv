#!/usr/bin/env bash
# Cozy gui to update the parameters for SWT

export png=${1%.*}.png
export out=/tmp/ccv-swt.png
swt_img='bash -c "run_swt -a %2 -b %3 -c %4 -d %5 -e %6 -f %7 -g %9 -h %10 -i %11 -j %13 -k %14 -l %15 -m %16 -n %17 -o %18 -p %20 -q %21 -r %22 -s %23 -t %24 -u %25 -v %26 -w %28 -x %29 " '
default_values='@bash -c "echo -e \"2:1\n3:1\n4:0\n5:1\n6:0.1\n7:0.8\n9:3\n10:124\n11:204\n13:300\n14:8\n15:38\n16:3\n17:8\n18:0.83\n20:1.5\n21:1.7\n22:31\n23:3\n24:2.9\n25:1.3\n26:1\n28:1\n29:1\" "'

run_swt() {
    local rect=
    #yad "double" is comma separated not c dot separated
    local com=(./swtdetect-gui ${*//,/.} $png)

    local i=1
    while IFS= read -r line; do
        par=($line)
        x=${par[0]}
        [[ $x == total ]] && break
        y=${par[1]}
        width=$((${par[2]}+x))
        height=$((${par[3]}+y))
        echo "$((i++)) ($x, $y) ($width, $height)"
        rect+=" -draw \"rectangle $x,$y,$width,$height\" "
    done < <("${com[@]}" 2>&1)

    rm -f $out
    eval convert $png -fill none -stroke yellow -strokewidth 3 $rect  $out 
    [[ -e $out ]] && xdg-open $out
}

export -f run_swt 

main() {
    [[ $# != 1 ]] && { echo "needs an image "; return ; }

    #nicer form with yad > 0.34.1 
    yad3.5 --form  --columns=5 \
        --field="<b>General options</b>":LBL ''\
        --field="interval:num" 1!0..10!1!0 \
        --field="min neighbors":NUM 1!0..10!1!0 \
        --field="scale invariant":NUM 0!0..1!1!0 \
        --field="direction":NUM 1!0..1!1!0 \
        --field="same word thr1":NUM 0.1!0..0.9!0.1!1 \
        --field="same word thr2":NUM 0.8!0..0.9!0.1!1 \
        --field="<b>Canny edge</b>":LBL ''  \
        --field="size":NUM 3!0..10!1!0 \
        --field="low thr":NUM 124!0..1000!1!0 \
        --field="high thr":NUM 204!0..1000!1!0 \
        --field="<b>Geometry filtering</b>":LBL ''\
        --field="max height":NUM 300!0..3000!1!0 \
        --field="min height":NUM 8!0..100!1!0 \
        --field="min area":NUM 38!0..100!1!0 \
        --field="leter occlude thr":NUM 3!0..10!1!0 \
        --field="aspect ratio":NUM 8!0..100!0.1!1 \
        --field="std ratio":NUM 0.83!0..1!0.1!2 \
        --field="<b>Grouping parameters</b>":LBL ''\
        --field="thickness ratio":NUM 1.5!0..10!0.1!1 \
        --field="height ratio":NUM 1.7!0..10!0.1!1 \
        --field="intensity thr":NUM 31!0..100!1!0 \
        --field="letter thr":NUM 3!0..10!1!0 \
        --field="distance ratio":NUM 2.9!0..10!0.1!1 \
        --field="intersect ratio":NUM 1.3!0..10!0.1!1 \
        --field="elongate ratio":NUM 1.9!0..1!0.1!1 \
        --field="<b>Break into words</b>":LBL ''\
        --field="Breakdown":NUM 1!0..1!1!0 \
        --field="breakdown ratio":NUM 1.0!0..10!0.1!1  \
        --field="SWT":FBTN "$swt_img" \
        --field="Defaults":FBTN "$default_values" \
        --no-buttons
}

main $@  


