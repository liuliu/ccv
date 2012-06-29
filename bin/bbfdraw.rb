#!/usr/bin/env ruby

exit unless ARGV.length == 2

rect = ""
STDIN.each_line do |line|
	print line
	args = line.split(" ")
	break if args[0] == 'total'
	x = args[0].to_i
	y = args[1].to_i
	width = args[2].to_i
	height = args[3].to_i
	rect += sprintf("-draw \"ellipse %d,%d %d,%d 0,360\" ", x + width / 2, y + height / 2, width / 2, height / 2)
end

%x[#{sprintf("convert %s -fill none -stroke green -strokewidth 2 %s%s", ARGV[0], rect, ARGV[1])}]
