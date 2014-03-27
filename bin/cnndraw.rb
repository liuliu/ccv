#!/usr/bin/env ruby

exit unless ARGV.length == 3

words = File.new ARGV[0]
labels = Hash.new
i = 1

words.each_line do |line|
	word = line.split ","
	labels[i.to_s] = word[0]
	i = i + 1
end

draw = ""
y = 15
STDIN.each_line do |line|
	print line
	args = line.split " "
	break if args[0] == 'elapsed'
	for i in 0..(args.length / 2 - 1)
		draw += sprintf("-fill none -strokewidth 1 -stroke DodgerBlue -draw \"rectangle 15,%d,165,%d\" -fill DodgerBlue -draw \"rectangle 15,%d,%d,%d\" -strokewidth 0 -stroke none -fill red -draw 'text 18,%d \"%s\"' ", y, y + 16, y, (args[i * 2 + 1].to_f * 150).to_i + 15, y + 16, y + 13, labels[args[i * 2]])
		y += 31
	end
end

%x[#{sprintf("convert %s %s%s", ARGV[1], draw, ARGV[2])}]
