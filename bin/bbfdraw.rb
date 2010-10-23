#!/usr/bin/env ruby

exit unless ARGV.length == 2

rect = ""
STDIN.each_line do |line|
	print line
	args = line.split(" ")
	break if args[0] == 'total'
	x = args[0].to_f
	y = args[1].to_f
	width = args[2].to_f
	height = args[3].to_f
	rect += sprintf("-draw \"rectangle %d,%d,%d,%d\" ", x, y, x + width, y + height)
end

%x[#{sprintf("convert %s -fill none -stroke red -strokewidth 3 %s%s", ARGV[0], rect, ARGV[1])}]
