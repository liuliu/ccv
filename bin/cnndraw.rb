#!/usr/bin/env ruby

exit unless ARGV.length == 3

words = File.new ARGV[0]
labels = Hash.new

words.each_line do |line|
	word = line.split " "
	first_word = (word[1, word.length - 1].join " ").split ","
	labels[word[0].to_s] = first_word[0]
end

draw = ""
y = 15
STDIN.each_line do |line|
	print line
	args = line.split " "
	break if args[0] == 'elapsed'
	draw += sprintf("-fill none -strokewidth 1 -stroke DodgerBlue -draw \"rectangle 15,%d,165,%d\" -fill DodgerBlue -draw \"rectangle 15,%d,%d,%d\" -strokewidth 0 -stroke none -fill red -draw 'text 18,%d \"%s\"' ", y, y + 16, y, (args[1].to_f * 150).to_i + 15, y + 16, y + 13, labels[args[0]])
	y += 31
end

%x[#{sprintf("convert %s %s%s", ARGV[1], draw, ARGV[2])}]
