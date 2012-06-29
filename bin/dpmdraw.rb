#!/usr/bin/env ruby

exit unless ARGV.length == 2

rect = ""
STDIN.each_line do |line|
	print line
	args = line.split(" ")
	break if args[0] == 'total'
	if args[0] == '|'
		x = args[1].to_i
		y = args[2].to_i
		width = args[3].to_i
		height = args[4].to_i
		rect += sprintf("-stroke \"#0000ff80\" -draw \"rectangle %d,%d,%d,%d\" ", x, y, x + width, y + height)
	else
		x = args[0].to_i
		y = args[1].to_i
		width = args[2].to_i
		height = args[3].to_i
		rect += sprintf("-stroke red -draw \"rectangle %d,%d,%d,%d\" ", x, y, x + width, y + height)
	end
end

%x[#{sprintf("convert %s -fill none -strokewidth 1 %s%s", ARGV[0], rect, ARGV[1])}]
