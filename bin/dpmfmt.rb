#!/usr/bin/env ruby

exit unless ARGV.length == 1

files = Dir.glob(ARGV[0] + '/*.txt')

files.each do |file|
	name = nil;
	boxes = Array.new
	File.new(file).each_line do |line|
		next if line[0] == '#'
		name = line[line.rindex('/') + 1, line.rindex('"') - (line.rindex('/') + 1)] if line[0, 14].downcase  == "image filename"
		if line[0, 16].downcase == "bounding box for"
			i = line.scan(/object\s*(\d+)/)[0][0].to_i
			coord = line.scan(/\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)/)[0]
			boxes[i - 1] = { :x => coord[0].to_i, :y => coord[1].to_i, :width => coord[2].to_i - coord[0].to_i, :height => coord[3].to_i - coord[1].to_i }
		end
	end
	boxes.each { |coord| print name + " " + coord[:x].to_s + " " + coord[:y].to_s + " " + coord[:width].to_s + " " + coord[:height].to_s + "\n" }
end
